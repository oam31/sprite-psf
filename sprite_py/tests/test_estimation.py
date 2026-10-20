import numpy as np

from conftest import assert_rel
from sprite import coadd, estimation
from sprite.wavelets import Undecimated79


def test_noise_sigma(units, cube):
    sigma = estimation.noise_sigma(cube)
    assert np.allclose(sigma, units.txt("noise_est.txt"), rtol=1e-6)


def test_centroids_and_flux(units, cube):
    sigma = estimation.noise_sigma(cube)
    psf_sig = float(units.txt("sig_lr.txt"))
    cents = estimation.estimate_centroids(cube, sigma, psf_sig)
    assert np.abs(cents - units.centroids).max() < 1e-5
    assert np.abs(2 * (cents - cents[0]) - units.shifts).max() < 1e-5
    flux = estimation.estimate_flux(cube, cents, 4 * psf_sig, cpp_compat=True)
    assert np.allclose(flux, units.txt("flux.txt"), rtol=1e-6)


def test_flux_aperture_fix():
    """Same source at different sub-pixel positions: the centred aperture sees equal fluxes."""
    yy, xx = np.indices((40, 40))
    centers = [(20.0, 20.0), (20.4, 19.7), (19.6, 20.45)]
    frames = np.array([np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2)) for cy, cx in centers])
    cents = np.array(centers) + 1.0  # 1-based, as returned by estimate_centroids
    fixed = estimation.estimate_flux(frames, cents, 3.0, cpp_compat=False)
    cpp = estimation.estimate_flux(frames, cents, 3.0, cpp_compat=True)
    assert np.abs(fixed - 1).max() < 0.01
    assert np.abs(cpp - 1).max() > 3 * np.abs(fixed - 1).max()


def test_coadd(units, cube):
    assert_rel(coadd.coadd(cube, 2), units.image("lr_psf_coadd.fits"), 1e-6)


def test_psf_sigma(units, cube):
    # Least-squares Gaussian fit instead of the C++ alternating Newton scheme.
    sig = coadd.psf_sigma(cube, 2, Undecimated79(6))
    assert abs(sig / units.txt("sig_lr.txt") - 1) < 1e-3


def test_gaussian_fit_on_cpp_model(units):
    # C++ (sigx, sigy) refer to its own (first, second) axes, i.e. natural (x, y).
    sy, sx = estimation.gaussian_fit(units.image("lr_psf_model.fits"))[:2]
    sig_x, sig_y = units.txt("lr_psf_gauss_fit.txt")[:2]
    assert np.allclose([sy, sx], [sig_y, sig_x], rtol=1e-3)


def test_fit_flux_exact_on_model():
    """Images exactement égales à f_k D H_k x : l'ajustement retrouve les f_k."""
    from sprite.operators import SpriteOperator

    yy, xx = np.indices((64, 64))
    model = np.exp(-((yy - 32.3) ** 2 + (xx - 31.6) ** 2) / (2 * 2.5 ** 2))
    shifts = np.array([[0.0, 0.0], [0.7, -0.4], [-0.3, 0.9]])
    flux = np.array([1.0, 0.85, 1.2])
    frames = SpriteOperator(shifts, flux, np.ones(3), (32, 32)).op(model)
    assert np.allclose(estimation.fit_flux(frames, model, shifts, 2), flux, rtol=1e-10)
