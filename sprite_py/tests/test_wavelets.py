import numpy as np
import pytest

from conftest import assert_rel
from sprite.wavelets import Starlet, Undecimated79, isap_median, mad_sigma

# The C++ stores float32 values: 1e-6 relative is the float32 noise floor here.
RTOL = 1e-6


def test_isap_median_is_lower_median():
    assert isap_median([4.0, 1.0, 3.0, 2.0]) == 2.0
    assert isap_median([5.0, 1.0, 3.0]) == 3.0


def test_mad_sigma_gaussian():
    x = np.random.default_rng(0).normal(scale=2.0, size=200_000)
    assert abs(mad_sigma(x) / 2.0 - 1) < 0.01


@pytest.mark.parametrize("transform", [Starlet(6), Undecimated79(4), Undecimated79(3, norm="l2")])
def test_perfect_reconstruction(transform):
    # iSAP's mirror border breaks exact 7/9 inversion once the dilated filter
    # (8 * 2**(n_scales - 2) + 1 taps) exceeds about twice the image size.
    x = np.random.default_rng(1).normal(size=(64, 48))
    assert np.allclose(transform.recons(transform.transform(x)), x, atol=1e-12)


def test_starlet_matches_isap(units):
    # Coefficients of an iSAP Ifloat image, taken in the same (C++) orientation.
    x = units.fits("rnd_hr.fits")
    coeffs = Starlet(6).transform(x)
    assert_rel(coeffs, units.fits("wt_sr.fits"), RTOL)
    assert_rel(Starlet(6).recons(coeffs), units.fits("wt_sr_rec.fits"), RTOL)


def test_undecimated79_matches_isap(units):
    x = units.fits("rnd_hr.fits")
    t = Undecimated79(6)
    coeffs = t.transform(x)
    assert_rel(coeffs, units.fits("wt_mod.fits"), RTOL)
    assert_rel(t.recons(coeffs), units.fits("wt_mod_rec.fits"), RTOL)


def test_undecimated79_l2_matches_isap(units):
    lr0 = units.fits("rnd_lr.fits")[0].T
    assert_rel(Undecimated79(2, norm="l2").transform(lr0), units.fits("wt_den.fits"), RTOL)


def test_band_noise_matches_isap(units):
    coeffs = Starlet(6).transform(units.fits("rnd_hr.fits"))
    ref = units.fits("wt_sr_gnoise.fits")
    for b in range(ref.shape[0]):
        assert abs(mad_sigma(coeffs[b]) / ref[b, 0, 0] - 1) < 1e-5


@pytest.mark.parametrize("transform,shape", [(Starlet(6), (64, 48)), (Undecimated79(4), (40, 52)),
                                             (Undecimated79(6), (168, 168))])
def test_exact_adjoint(transform, shape):
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape)
    c = rng.normal(size=(transform.n_bands,) + shape)
    assert abs(np.vdot(transform.transform(x), c) / np.vdot(x, transform.adjoint(c)) - 1) < 1e-12
