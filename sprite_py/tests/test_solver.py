"""End-to-end comparison with run_sprite -s 4 -r 2 -F -N (paper settings)."""

import numpy as np
import pytest

from conftest import assert_rel
from sprite import sprite

# cpp_compat=True: these tests check the port against the C++ itself.
PAPER = dict(kappa=4, estimate_flux=True, estimate_noise_per_frame=True, cpp_compat=True)


def _run(cube, ref_units, transform, **kwargs):
    return sprite(
        cube, transform=transform,
        spectral_radius=ref_units.txt("spectral_rad.txt")[0],
        psf_sigma=float(ref_units.txt("sig_lr.txt")),
        **PAPER, **kwargs,
    )


def test_starlet_matches_cpp(cube, units, golden_root):
    res = _run(cube, units, "starlet")
    assert_rel(res.solution, golden_root.image("t2/solution.fits"), 1e-5)
    assert_rel(res.first_guess, golden_root.image("t2/model.fits"), 1e-6)
    # Le C++ écrit données - résidu de la dernière itération ; lr_model est prédit par la solution finale.
    assert_rel(cube - res.residual, golden_root.fits("t2/lr_solution.fits"), 1e-5)
    assert_rel(res.lr_model, golden_root.fits("t2/lr_solution.fits"), 1e-3)
    assert_rel(res.residual, golden_root.fits("t2/resi.fits"), 1e-3)
    mse = golden_root.txt("t2/mse.txt")  # last reweighting pass only
    assert np.allclose(res.residual_energy[-len(mse):], mse, rtol=1e-3)
    params = (golden_root.path / "t2/param.txt").read_text()
    n_cv = int(params.split("Number of iterations for convergence:")[1].split()[0])
    assert res.n_iter == n_cv


@pytest.mark.slow
def test_undecimated79_matches_cpp(cube, units_t24, golden_root):
    res = _run(cube, units_t24, "u79")
    assert_rel(res.solution, golden_root.image("t24/solution.fits"), 1e-5)


def test_standalone_pipeline_close_to_cpp(cube, golden_root):
    """No C++ value injected: only the spectral-radius start vector differs."""
    res = sprite(cube, transform="starlet", **PAPER)
    assert_rel(res.solution, golden_root.image("t2/solution.fits"), 5e-4)
