"""Chemins d'utilisation non couverts ailleurs : ligne de commande, options du solveur, Bayer, entrées-sorties."""

import json

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from sprite import SpriteConfig, sprite
from sprite.bayer import refine_pose_shifts, sprite_bayer, sublattices
from sprite.cli import check_main, main
from sprite.io import load_bad_pixels, load_frames, save_image
from sprite.masking import build_masks, summary
from sprite.operators import SpriteOperator
from sprite.shifts import estimate_shifts, fourier_shift, refine_on_model


def _frames(n=8, sigma=2e-3, seed=0, flux=None):
    """Petites images d'une gaussienne elliptique, décalages connus (pixels BR)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((64, 64))
    psf = np.exp(-((yy - 32.2) ** 2 / (2 * 1.6 ** 2) + (xx - 31.7) ** 2 / (2 * 2.0 ** 2)))
    psf /= psf.sum()
    shifts = np.vstack([[0.0, 0.0], rng.uniform(-1, 1, size=(n - 1, 2))])
    flux = np.ones(n) if flux is None else np.asarray(flux)
    clean = SpriteOperator(shifts, flux, np.ones(n), (32, 32)).op(psf)
    return clean + rng.normal(scale=sigma * clean.max(), size=clean.shape), shifts / 2, psf


# --------------------------------------------------------------------------- solveur

def test_solver_options_paths():
    frames, lr_shifts, _ = _frames()
    base = dict(estimate_noise_per_frame=True)
    r = sprite(frames, shift_refine=1, **base)
    assert len(r.shift_history) == 2
    r = sprite(frames, shift_method="xcorr", **base)
    assert r.shifts.shape == (8, 2)
    r = sprite(frames, estimate_flux=True, flux_method="fit", flux_refine=1, **base)
    assert abs(r.flux[0] - 1) < 1e-12 and np.all(np.abs(r.flux - 1) < 0.1)
    r = sprite(frames, fluxes=2 * np.ones(8), **base)
    assert np.allclose(r.flux, 1.0)
    r = sprite(frames, shifts=lr_shifts, **base)
    assert np.allclose(r.shifts, 2 * lr_shifts)
    with pytest.raises(ValueError, match="méthode de flux"):
        sprite(frames, estimate_flux=True, flux_method="inconnue", **base)
    with pytest.raises(TypeError, match="inconnu"):
        sprite(frames, parametre_inexistant=1)


def test_flux_fit_recovers_unequal_fluxes():
    flux = np.array([1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.85, 1.15])
    frames, _, _ = _frames(flux=flux, sigma=1e-3)
    r = sprite(frames, estimate_noise_per_frame=True, estimate_flux=True, flux_method="fit", flux_refine=2)
    assert np.abs(r.flux - flux).max() < 0.05


# --------------------------------------------------------------------------- décalages

def test_refine_on_model_exact_with_true_model():
    frames, lr_shifts, psf = _frames(sigma=1e-4)
    noisy_init = lr_shifts + np.random.default_rng(3).normal(scale=0.05, size=lr_shifts.shape)
    sigma = np.full(len(frames), 1e-4 * frames.max())
    valid = np.ones(frames.shape, bool)
    shifts, flux = refine_on_model(frames, sigma, psf, 2, noisy_init, np.ones(len(frames)), valid=valid)
    assert np.abs(shifts - lr_shifts).max() < 0.01
    with pytest.raises(ValueError, match="méthode de décalage"):
        estimate_shifts(frames, sigma, 1.0, method="inconnue")


# --------------------------------------------------------------------------- Bayer

def _mosaic(n=16, seed=1, pattern="RGGB"):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((64, 64))
    base = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 2.2 ** 2))
    truth = {"R": fourier_shift(base, (0.0, 1.0)), "G": base, "B": base}
    pose = np.vstack([[0.0, 0.0], rng.uniform(-1.5, 1.5, size=(n - 1, 2))])
    mosaic = np.zeros((n, 32, 32))
    for k, s in enumerate(pose):
        for b, offs in sublattices(pattern).items():
            full = fourier_shift(truth[b], 2 * s)[::2, ::2]
            for oy, ox in offs:
                mosaic[k, oy::2, ox::2] = full[oy::2, ox::2]
    return truth, mosaic + 50.0 + rng.normal(scale=1e-3, size=mosaic.shape), pose


def test_refine_pose_shifts_exact_with_true_models():
    truth, mosaic, pose = _mosaic()
    init = pose + np.random.default_rng(2).normal(scale=0.05, size=pose.shape)
    init -= init[0]
    refined = refine_pose_shifts(mosaic - 50.0, truth, init, 2)
    assert np.sqrt(np.mean((refined - pose) ** 2)) < 0.01


def test_sprite_bayer_options():
    truth, mosaic, pose = _mosaic(pattern="BGGR")
    bad = np.zeros(mosaic.shape[1:], bool)
    bad[2, 3] = True
    res = sprite_bayer(mosaic, pattern="BGGR", subtract_background=True, bad_pixels=bad,
                       pose_refine=1, fluxes=np.ones(len(mosaic)), estimate_noise_per_frame=True)
    assert set(res.bands) == {"R", "G", "B"}
    assert res.bands["G"].solution.shape == (64, 64)
    assert "Décalage R-G" in res.text()
    assert np.sqrt(np.mean((res.pose_shifts - pose) ** 2)) < 0.3


# --------------------------------------------------------------------------- ligne de commande

def test_cli_diagnostics_and_shift_file(tmp_path):
    frames, lr_shifts, _ = _frames()
    fits.writeto(tmp_path / "cube.fits", frames.astype(np.float32))
    np.savetxt(tmp_path / "shifts.txt", lr_shifts[:, ::-1])  # colonnes x puis y
    out = tmp_path / "out"
    main(["-N", "-W", "-w", "resi.fits", "-S", str(tmp_path / "shifts.txt"), "--cpp-orientation",
          str(tmp_path / "cube.fits"), "sol.fits", str(out)])
    for f in ("sol.fits", "lr_sol.fits", "resi.fits", "model.fits", "pixel_map.fits", "flux.txt",
              "shift.txt", "sig_est.txt", "mse.txt", "param.txt", "centroids.txt"):
        assert (out / f).exists(), f
    assert np.allclose(np.loadtxt(out / "shift.txt"), 2 * lr_shifts[:, ::-1], atol=1e-5)
    assert "Analysis prox: exact adjoint" in (out / "param.txt").read_text()


def test_cli_bayer(tmp_path):
    _, mosaic, pose = _mosaic()
    for k, m in enumerate(mosaic):
        tifffile.imwrite(tmp_path / f"p{k:02d}.tif", m.astype(np.float32))
    np.savetxt(tmp_path / "pose.txt", pose[:, ::-1])
    out = tmp_path / "out"
    main(["-N", "-W", "--bayer", "RGGB", "--background", "-S", str(tmp_path / "pose.txt"),
          str(tmp_path / "p*.tif"), "psf.tif", str(out)])
    for f in ("psf_R.tif", "psf_G.tif", "psf_B.tif", "chromatique.txt", "pose_shift.txt", "flux_G.txt"):
        assert (out / f).exists(), f
    assert tifffile.imread(out / "psf_G.tif").shape == (64, 64)


def test_cli_check_and_scan(tmp_path, capsys):
    frames, _, _ = _frames()
    fits.writeto(tmp_path / "cube.fits", frames.astype(np.float32))
    check_main(["-N", "--folds", "2", "--json", str(tmp_path / "c.json"), str(tmp_path / "cube.fits")])
    assert "Contrôle de la reconstruction" in capsys.readouterr().out
    assert "holdout_chi2" in json.loads((tmp_path / "c.json").read_text())
    check_main(["-N", "--folds", "2", "--scan", "kappa", "3", "5", "--json", str(tmp_path / "s.json"),
                str(tmp_path / "cube.fits")])
    text = capsys.readouterr().out
    assert "retenue" in text and json.loads((tmp_path / "s.json").read_text())["scan"] == "kappa"
    with pytest.raises(SystemExit):
        check_main(["--scan", "kappa", "3", str(tmp_path / "cube.fits")])


def test_cli_check_bayer(tmp_path, capsys):
    _, mosaic, pose = _mosaic()
    for k, m in enumerate(mosaic):
        tifffile.imwrite(tmp_path / f"p{k:02d}.tif", m.astype(np.float32))
    np.savetxt(tmp_path / "pose.txt", pose[:, ::-1])
    check_main(["-N", "--bayer", "RGGB", "--background", "--folds", "2", "-j", "2", "-S",
                str(tmp_path / "pose.txt"), "--json", str(tmp_path / "b.json"), str(tmp_path / "p*.tif")])
    assert "mosaïque de Bayer" in capsys.readouterr().out
    data = json.loads((tmp_path / "b.json").read_text())
    assert set(data["bands"]) == {"R", "G", "B"} and "R-G" in data["chromatic_split"]
    with pytest.raises(SystemExit):
        check_main(["--bayer", "RGGB", "--scan", "kappa", "3", "5", str(tmp_path / "p*.tif")])


# --------------------------------------------------------------------------- entrées-sorties et masques

def test_io_bad_pixels_and_errors(tmp_path):
    bad2d = np.zeros((8, 8), np.uint16)
    bad2d[1, 2] = 1
    tifffile.imwrite(tmp_path / "bad.tif", bad2d)
    assert load_bad_pixels(tmp_path / "bad.tif").shape == (8, 8)
    tifffile.imwrite(tmp_path / "bad3.tif", np.stack([bad2d, bad2d]))
    assert load_bad_pixels(tmp_path / "bad3.tif").shape == (2, 8, 8)
    save_image(tmp_path / "x.fits", np.ones((4, 4)))
    assert fits.getdata(tmp_path / "x.fits").dtype == np.dtype(">f4")
    (tmp_path / "a.png").write_bytes(b"")
    with pytest.raises(ValueError, match="format"):
        load_frames(tmp_path / "a.png")
    with pytest.raises(ValueError, match="au moins deux"):
        load_frames(tmp_path / "x.fits")
    with pytest.raises(FileNotFoundError):
        load_frames(str(tmp_path / "rien*.tif"))


def test_masks_summary():
    frames = np.zeros((2, 6, 6))
    frames[0, 2, 2] = 10.0
    bad = np.zeros((6, 6), bool)
    bad[0, 0] = True
    valid, sat = build_masks(frames, saturation=5.0, bad_pixels=bad)
    assert summary(valid, sat) == "pixels saturés : 1, 0 ; exclus : 1, 1"
