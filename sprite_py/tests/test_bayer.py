"""Reconstruction par bande sur mosaïque de Bayer."""

import numpy as np
import pytest

from sprite.bayer import lattice_shifts, split_bayer, sprite_bayer, sublattices
from sprite.shifts import fourier_shift


def test_sublattices_and_split():
    assert sublattices("RGGB") == {"R": [(0, 0)], "G": [(0, 1), (1, 0)], "B": [(1, 1)]}
    assert sublattices("BGGR")["R"] == [(1, 1)]
    frames = np.arange(2 * 4 * 6, dtype=float).reshape(2, 4, 6)
    parts = split_bayer(frames)
    sub, off = parts["B"][0]
    assert off == (1, 1) and np.array_equal(sub, frames[:, 1::2, 1::2])
    with pytest.raises(ValueError):
        split_bayer(np.zeros((2, 5, 6)))


def test_lattice_shift_convention():
    assert np.allclose(lattice_shifts(np.array([[0.0, 0.0], [1.0, -2.0]]), (0, 1)), [[0.0, -0.5], [0.5, -1.5]])


def _mosaic(n=16, seed=1, lateral=(0.0, 1.0)):
    """PSF gaussiennes par bande ; R décalée de `lateral` pixels HR par rapport à G et B."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((64, 64))
    base = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 2.2 ** 2))
    truth = {"R": fourier_shift(base, lateral), "G": base, "B": base}
    pose = np.vstack([[0.0, 0.0], rng.uniform(-1, 1, size=(n - 1, 2))])
    mosaic = np.zeros((n, 32, 32))
    for k, s in enumerate(pose):
        for b, offs in sublattices("RGGB").items():
            full = fourier_shift(truth[b], 2 * s)[::2, ::2]
            for oy, ox in offs:
                mosaic[k, oy::2, ox::2] = full[oy::2, ox::2]
    mosaic += rng.normal(scale=1e-3, size=mosaic.shape)
    return truth, mosaic, pose


def test_bayer_reconstruction_with_true_shifts():
    truth, mosaic, pose = _mosaic()
    res = sprite_bayer(mosaic, pose_shifts=pose, estimate_noise_per_frame=True)
    # G (deux sous-réseaux) est mieux contraint que R et B (un sous-réseau chacun).
    for b, tol in (("R", 0.25), ("G", 0.05), ("B", 0.25)):
        sol, ref = res.bands[b].solution, truth[b]
        assert sol.shape == ref.shape
        assert np.linalg.norm(sol / sol.sum() - ref / ref.sum()) / np.linalg.norm(ref / ref.sum()) < tol
    # décalage chromatique latéral de R retrouvé (1 pixel HR en x)
    assert abs(res.chromatic["R-G"][1] - 1.0) < 0.15 and abs(res.chromatic["R-G"][0]) < 0.15
