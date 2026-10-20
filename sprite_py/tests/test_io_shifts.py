"""TIFF/FITS input and shift estimators."""

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from sprite import estimation
from sprite.cli import main
from sprite.io import border_background, load_frames
from sprite.shifts import estimate_shifts, fourier_shift, xcorr_shift


def _gauss_frames(n=4, size=40, seed=0, s=2.0):
    """Well-sampled Gaussian frames with known shifts (dy, dx), LR pixels."""
    rng = np.random.default_rng(seed)
    shifts = np.vstack([[0.0, 0.0], rng.uniform(-0.5, 0.5, size=(n - 1, 2))])
    yy, xx = np.indices((size, size))
    c = size / 2
    frames = np.array([np.exp(-((yy - c - dy) ** 2 + (xx - c - dx) ** 2) / (2 * s ** 2)) for dy, dx in shifts])
    return frames, shifts


@pytest.fixture
def frames_u16():
    frames, _ = _gauss_frames()
    return np.round(1000 + 30000 * frames).astype(np.uint16)


def test_multipage_tiff_uint16(tmp_path, frames_u16):
    path = tmp_path / "cube.tif"
    tifffile.imwrite(path, frames_u16)
    frames, dtypes = load_frames(path)
    assert dtypes == ["uint16"]
    assert frames.dtype == np.float64 and np.array_equal(frames, frames_u16)


def test_one_float32_tiff_per_frame(tmp_path, frames_u16):
    data = frames_u16.astype(np.float32) / 7.0
    for k, f in enumerate(data):
        tifffile.imwrite(tmp_path / f"frame_{k}.tif", f)
    for inputs in (tmp_path, str(tmp_path / "frame_*.tif"),
                   [tmp_path / f"frame_{k}.tif" for k in range(4)]):
        frames, dtypes = load_frames(inputs)
        assert set(dtypes) == {"float32"}
        assert np.allclose(frames, data)


def test_bands_as_samples(tmp_path, frames_u16):
    path = tmp_path / "bands.tif"
    tifffile.imwrite(path, np.moveaxis(frames_u16, 0, -1), planarconfig="contig")
    frames, _ = load_frames(path)
    assert np.array_equal(frames, frames_u16)


def test_fits_cube_and_mismatch(tmp_path, frames_u16):
    fits.writeto(tmp_path / "cube.fits", frames_u16.astype(np.float32))
    assert load_frames(tmp_path / "cube.fits")[0].shape == (4, 40, 40)
    tifffile.imwrite(tmp_path / "small.tif", frames_u16[0, :20, :20])
    with pytest.raises(ValueError, match="tailles différentes"):
        load_frames([tmp_path / "cube.fits", tmp_path / "small.tif"])


def test_border_background(frames_u16):
    assert np.allclose(border_background(frames_u16.astype(float)), 1000, atol=1)


def test_xcorr_and_fourier_shift_sign():
    rng = np.random.default_rng(3)
    ref = np.fft.irfft2(np.fft.rfft2(rng.normal(size=(64, 64))) * np.exp(-np.fft.rfftfreq(64)[None] ** 2 * 200
                                                                         - np.fft.fftfreq(64)[:, None] ** 2 * 200), s=(64, 64))
    moved = fourier_shift(ref, (0.37, -1.21))
    assert np.allclose(xcorr_shift(ref, moved), (0.37, -1.21), atol=0.02)


@pytest.mark.parametrize("method", ["centroid_wide", "xcorr"])
def test_shift_methods_on_sampled_gaussians(method):
    frames, shifts = _gauss_frames()
    sigma = np.full(len(frames), 1e-3)
    est, _ = estimate_shifts(frames, sigma, psf_sigma=2.0, method=method)
    assert np.abs(est - shifts).max() < 0.02


def test_cli_with_tiff(tmp_path, cube):
    for k, f in enumerate(cube):
        tifffile.imwrite(tmp_path / f"f{k}.tif", f.astype(np.float32))
    out = tmp_path / "out"
    main(["-t", "2", "-s", "4", "-N", "-W", "--shift-method", "centroid_wide",
          str(tmp_path / "f*.tif"), "psf.tif", str(out)])
    sol = tifffile.imread(out / "psf.tif")
    assert sol.shape == (168, 168) and sol.dtype == np.float32 and sol.max() > 0
    assert (out / "lr_psf.tif").exists() and (out / "shift.txt").exists()


def test_default_shift_method():
    from sprite import SpriteConfig

    assert SpriteConfig().resolved_shift_method() == "centroid_wide"
    assert SpriteConfig(cpp_compat=True).resolved_shift_method() == "centroid"
    assert SpriteConfig(cpp_compat=True, shift_method="xcorr").resolved_shift_method() == "xcorr"


def test_default_exact_adjoint():
    from sprite import SpriteConfig

    assert SpriteConfig(transform="starlet").resolved_exact_adjoint() is True
    assert SpriteConfig(transform="u79").resolved_exact_adjoint() is False
    assert SpriteConfig(transform="starlet", cpp_compat=True).resolved_exact_adjoint() is False
    assert SpriteConfig(transform="u79", exact_adjoint=True).resolved_exact_adjoint() is True


def test_starlet_has_no_background(cube):
    """Avec l'adjoint exact, la starlet ne laisse plus de fond étendu (défaut du C++)."""
    from sprite import sprite

    sol = sprite(cube, transform="starlet", kappa=4, estimate_noise_per_frame=True).solution
    border = np.median(np.r_[sol[:20].ravel(), sol[-20:].ravel()])
    assert border < 1e-3 * sol.max()
