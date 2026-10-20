"""Pixels saturés ou défectueux."""

import numpy as np
import pytest

from sprite import sprite
from sprite.io import border_background
from sprite.masking import build_mask, inpaint


def test_build_mask_sources():
    frames = np.zeros((2, 6, 6))
    frames[0, 1, 1] = 100.0
    frames[1, 4, 4] = np.nan
    bad = np.zeros((6, 6), bool)
    bad[0, 5] = True
    valid = build_mask(frames, saturation=50.0, bad_pixels=bad)
    assert not valid[0, 1, 1] and not valid[1, 4, 4]
    assert not valid[0, 0, 5] and not valid[1, 0, 5]
    assert valid.sum() == 2 * 36 - 4
    dilated = build_mask(frames, saturation=50.0, dilate=1)
    assert (~dilated[0]).sum() == 9


def test_build_mask_shape_error():
    with pytest.raises(ValueError, match="pixels défectueux"):
        build_mask(np.zeros((2, 6, 6)), bad_pixels=np.zeros((5, 5), bool))


def test_inpaint_smooth_image():
    yy, xx = np.indices((20, 20))
    img = (0.3 * yy + 0.1 * xx)[None].astype(float)
    valid = np.ones_like(img, bool)
    valid[0, 8:11, 8:11] = False
    filled = inpaint(img, valid)
    assert np.allclose(filled[valid], img[valid])
    assert np.abs(filled - img).max() < 1e-6


def test_background_ignores_invalid_pixels():
    frames = np.full((1, 40, 40), 10.0)
    frames[0, :4, :] = 1e4  # bande chaude dans la bordure
    valid = frames < 1e3
    assert border_background(frames, valid=valid)[0] == 10.0


def _run(frames, **kw):
    return sprite(frames, transform="starlet", kappa=4, estimate_noise_per_frame=True,
                  n_iter=30, spectral_radius=4.85e9, **kw)


def test_masked_pixels_do_not_influence_solution(cube):
    """Deux jeux qui ne diffèrent que sur les pixels masqués donnent la même solution."""
    rng = np.random.default_rng(4)
    bad = np.zeros(cube.shape, bool)
    bad[:, 10:14, 60:64] = True  # loin de l'étoile : l'interpolation y est sans effet
    a = cube.copy()
    b = cube.copy()
    b[bad] = rng.normal(scale=1e-2, size=bad.sum())
    ra = _run(a, bad_pixels=bad, psf_sigma=0.735)
    rb = _run(b, bad_pixels=bad, psf_sigma=0.735)
    assert np.array_equal(ra.solution, rb.solution)


def test_mask_restores_corrupted_reconstruction(cube):
    """Pixels chauds et cœur saturé : le masque ramène la solution près de la solution propre."""
    rng = np.random.default_rng(5)
    clean = _run(cube).solution
    corrupted = cube.copy()
    hot = rng.random(cube.shape) < 0.003
    corrupted[hot] += 50 * 2.4e-5
    level = 0.6 * cube.max()
    corrupted = np.minimum(corrupted, level)
    raw = _run(corrupted).solution
    masked = _run(corrupted, saturation=level, bad_pixels=hot).solution
    err = lambda s: np.linalg.norm(s / s.sum() - clean / clean.sum()) / np.linalg.norm(clean / clean.sum())
    assert err(masked) < 0.5 * err(raw)
