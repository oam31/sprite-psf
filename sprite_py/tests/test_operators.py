import numpy as np

from conftest import assert_rel
from sprite.operators import (
    SpriteOperator, convolve, decimate, kernel_otf, lanczos_kernel, zero_insert,
)

RTOL = 1e-6


def _operator(units):
    return SpriteOperator(units.shifts, units.txt("flux.txt"), units.txt("noise_est.txt"), (84, 84))


def test_lanczos_kernel(units):
    # C++ lanczos(u = (0.3, -0.7)) along its (x, y) axes
    assert_rel(lanczos_kernel((-0.7, 0.3)), units.image("lanczos_03_m07.fits"), 1e-7)


def test_convolution_matches_psf_convol(units):
    x = units.image("rnd_hr.fits")
    otf = kernel_otf(lanczos_kernel((-0.7, 0.3)), x.shape)
    conv = convolve(x, otf)
    assert_rel(conv, units.image("conv_hr.fits"), RTOL)
    assert_rel(decimate(conv, 2), units.image("decim_hr.fits"), RTOL)


def test_zero_insert(units):
    lr0 = units.fits("rnd_lr.fits")[0]
    assert np.array_equal(zero_insert(lr0, 2), units.image("transpose_decim.fits"))


def test_registration_kernels(units):
    op = _operator(units)
    assert_rel(op.kernels, units.fits("lanczos_ker.fits"), 1e-6)


def test_forward_operator(units):
    op = _operator(units)
    assert_rel(op.op(units.image("rnd_hr.fits")), units.fits("op_M.fits"), RTOL)


def test_adjoint_operator(units):
    op = _operator(units)
    assert_rel(op.adj_op(units.fits("rnd_lr.fits")), units.image("op_MT.fits"), RTOL)


def test_adjoint_identity():
    rng = np.random.default_rng(2)
    shifts = rng.uniform(-1, 1, size=(5, 2))
    op = SpriteOperator(shifts, rng.uniform(0.5, 1.5, 5), rng.uniform(0.5, 2, 5), (32, 40), upfact=2)
    x = rng.normal(size=op.hr_shape)
    y = rng.normal(size=(5, 32, 40))
    assert abs(np.vdot(op.op(x), y) / np.vdot(x, op.adj_op(y)) - 1) < 1e-12


def test_spectral_radius(units):
    rho, _ = _operator(units).spectral_radius()
    # Different random start vector than the C++ rand(): agreement to the tolerance.
    assert abs(rho / units.txt("spectral_rad.txt")[0] - 1) < 1e-3
