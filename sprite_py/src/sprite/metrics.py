"""Mesures de forme d'une PSF (section 4.3 du papier).

Ces mesures servent à comparer une reconstruction à une référence (données
simulées) ou deux reconstructions entre elles (données réelles, voir
:mod:`sprite.diagnostics`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


def ellipticity(img: NDArray[np.float64], sigma_w: float = 4.0) -> tuple[float, float]:
    """Ellipticités par moments d'ordre 2 à pondération gaussienne (éq. 27 à 29 du papier).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        PSF.
    sigma_w : float, default 4.0
        Écart type de la pondération gaussienne, en pixels ; le centre de la
        pondération est itéré sur le centroïde pondéré.

    Returns
    -------
    e1 : float
        ``(mu20 - mu02) / (mu20 + mu02)``.
    e2 : float
        ``2 mu11 / (mu20 + mu02)``.
    """
    yy, xx = np.indices(img.shape)
    w = np.ones_like(img)
    cy, cx = np.array(img.shape) / 2
    for _ in range(10):
        wi = img * w
        cy, cx = (wi * yy).sum() / wi.sum(), (wi * xx).sum() / wi.sum()
        w = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma_w ** 2))
    wi = img * w
    m20 = (wi * (yy - cy) ** 2).sum()
    m02 = (wi * (xx - cx) ** 2).sum()
    m11 = (wi * (yy - cy) * (xx - cx)).sum()
    return float((m20 - m02) / (m20 + m02)), float(2 * m11 / (m20 + m02))


def fwhm(img: NDArray[np.float64], half: int = 12) -> float:
    """Largeur à mi-hauteur par ajustement d'une lorentzienne modifiée (profil de Moffat).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        PSF ; le pic doit être à plus de ``half`` pixels des bords.
    half : int, default 12
        Demi-taille de la fenêtre d'ajustement, en pixels.

    Returns
    -------
    float
        FWHM, en pixels de ``img``.
    """
    c = np.array(np.unravel_index(np.argmax(img), img.shape))
    sl = tuple(slice(ci - half, ci + half + 1) for ci in c)
    sub = img[sl]
    yy, xx = np.indices(sub.shape)

    def model(p: NDArray[np.float64]) -> NDArray[np.float64]:
        a, cy, cx, g, beta = p
        return a / (1 + ((yy - cy) ** 2 + (xx - cx) ** 2) / g ** 2) ** beta

    p0 = [sub.max(), half, half, 2.0, 2.0]
    res = least_squares(lambda p: (model(p) - sub).ravel(), p0,
                        bounds=([0, half / 2, half / 2, 0.3, 0.5], [np.inf, 1.5 * half, 1.5 * half, 20, 10]))
    _, _, _, g, beta = res.x
    return float(2 * g * np.sqrt(2 ** (1 / beta) - 1))


def compare(est: NDArray[np.float64], ref: NDArray[np.float64]) -> dict[str, float]:
    """Écarts de forme entre deux PSF, après normalisation à une somme unité.

    Parameters
    ----------
    est : ndarray, shape (ny, nx)
        PSF estimée.
    ref : ndarray, shape (ny, nx)
        PSF de référence.

    Returns
    -------
    dict
        ``rel_err`` (‖est − ref‖ / ‖ref‖), ``e1_err`` et ``e2_err`` (écarts
        absolus d'ellipticité), ``fwhm_err`` (écart relatif de FWHM).
    """
    est = est / est.sum()
    ref = ref / ref.sum()
    e_est, e_ref = ellipticity(est), ellipticity(ref)
    return {
        "rel_err": float(np.linalg.norm(est - ref) / np.linalg.norm(ref)),
        "e1_err": float(abs(e_est[0] - e_ref[0])),
        "e2_err": float(abs(e_est[1] - e_ref[1])),
        "fwhm_err": float(abs(fwhm(est) / fwhm(ref) - 1)),
    }
