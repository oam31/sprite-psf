"""Contrôles sur données réelles (sprite.diagnostics)."""

import numpy as np

from sprite.diagnostics import check
from sprite.operators import SpriteOperator


def _frames(n=8, sigma=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((64, 64))
    psf = np.exp(-((yy - 32.2) ** 2 / (2 * 1.6 ** 2) + (xx - 31.7) ** 2 / (2 * 2.0 ** 2)))
    psf /= psf.sum()
    shifts = np.vstack([[0.0, 0.0], rng.uniform(-1, 1, size=(n - 1, 2))])
    clean = SpriteOperator(shifts, np.ones(n), np.ones(n), (32, 32)).op(psf)
    return clean + rng.normal(scale=sigma * clean.max(), size=clean.shape)


def test_check_well_specified():
    frames = _frames()
    _, rep = check(frames, estimate_noise_per_frame=True, n_folds=2)
    assert 0.7 < np.median(rep.chi2) < 1.3
    assert np.median(rep.holdout_chi2) / np.median(rep.chi2) < 1.3
    assert rep.split["rel_err"] < 0.5
    assert "Contrôle de la reconstruction" in rep.text()


def test_scan_kappa_marks_best():
    from sprite import SpriteConfig
    from sprite.diagnostics import scan

    rows = scan(_frames(), SpriteConfig(estimate_noise_per_frame=True), "kappa", [2, 4], n_folds=2)
    assert len(rows) == 2 and sum(r["recommended"] for r in rows) == 1 and sum(r["minimum"] for r in rows) == 1
    assert all(r["holdout_chi2"] > 0 for r in rows)


def test_parallel_matches_sequential():
    from sprite.parallel import run_parallel

    assert run_parallel(divmod, [(7, 2), (9, 4), (5, 5)], n_jobs=2) == [(3, 1), (2, 1), (1, 0)]
    frames = _frames(seed=1)
    _, seq = check(frames, estimate_noise_per_frame=True, n_folds=2, n_jobs=1)
    _, par = check(frames, estimate_noise_per_frame=True, n_folds=2, n_jobs=3)
    assert np.array_equal(seq.holdout_chi2, par.holdout_chi2) and seq.split == par.split


def test_check_bayer_whole_poses():
    from sprite.bayer import sublattices
    from sprite.diagnostics import check_bayer
    from sprite.shifts import fourier_shift

    rng = np.random.default_rng(1)
    n = 16
    yy, xx = np.indices((64, 64))
    base = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 2.2 ** 2))
    pose = np.vstack([[0.0, 0.0], rng.uniform(-1.5, 1.5, size=(n - 1, 2))])
    mosaic = np.zeros((n, 32, 32))
    for k, s in enumerate(pose):
        full = fourier_shift(base, 2 * s)[::2, ::2]
        for offs in sublattices("RGGB").values():
            for oy, ox in offs:
                mosaic[k, oy::2, ox::2] = full[oy::2, ox::2]
    mosaic += rng.normal(scale=1e-3, size=mosaic.shape)
    full, rep = check_bayer(mosaic, pose_shifts=pose, estimate_noise_per_frame=True, n_folds=2)
    assert set(rep.bands) == {"R", "G", "B"}
    for r in rep.bands.values():
        assert r.chi2.shape == (n,) and r.holdout_chi2.shape == (n,) and np.all(np.isfinite(r.holdout_chi2))
        assert 0.3 < np.median(r.chi2) < 1.5  # bruit surestimé sur des sous-images de 16 x 16
    # Même PSF dans les trois bandes : décalages chromatiques nuls et stables.
    assert np.linalg.norm(full.chromatic["R-G"]) < 0.1 and rep.chromatic_split["B-G"] < 0.1
    assert any("16 poses" in note for note in rep.notes)
    text = rep.text()
    assert "mosaïque de Bayer" in text and "écart entre moitiés" in text
