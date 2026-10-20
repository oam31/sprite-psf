"""Estimation des décalages sous-pixel entre images BR.

Toutes les méthodes renvoient des décalages ``(dy, dx)`` en pixels BR,
relatifs à l'image de référence : l'image ``k`` est l'image de référence
déplacée de ``+décalage``.

* ``centroid`` : méthode du C++ (éq. 24 du papier) ; images seuillées,
  centroïde à pondération gaussienne de largeur ``psf_sigma`` pixels BR. Avec
  une fenêtre aussi étroite, l'estimation se verrouille sur la grille des
  pixels quand la PSF est sous-échantillonnée.
* ``centroid_wide`` (défaut) : même estimateur, fenêtre ``wide_factor`` fois
  plus large, ce qui supprime l'essentiel du verrouillage.
* ``xcorr`` : corrélation croisée avec l'image de référence, affinée au
  ``1/upsample`` de pixel par une DFT matricielle (Guizar-Sicairos, Thurman et
  Fienup 2008), sur images seuillées à ``nsig`` écarts types.

:func:`refine_on_model` ajuste décalage et flux de chaque image sur un modèle
HR, par moindres carrés (modèle décalé par Fourier puis décimé). Ce n'est
précis que face à un bon modèle : la solution SPRITE (option ``shift_refine``
du solveur), pas la pile de Lanczos d'images repliées, dont l'ajustement
absorbe les erreurs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .estimation import estimate_centroids, hard_threshold, weighted_centroid

METHODS = ("centroid", "centroid_wide", "xcorr")


def _upsampled_dft(
    data: NDArray[np.complex128], size: int, factor: float, offsets: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """DFT inverse de ``data`` sur une grille de ``size`` points de pas ``1/factor``."""
    for n, off in zip(data.shape[::-1], offsets[::-1]):
        kernel = np.exp(-2j * np.pi * (np.arange(size) - off)[:, None] * np.fft.fftfreq(n, factor)[None, :])
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def xcorr_shift(ref: NDArray[np.float64], img: NDArray[np.float64], upsample: int = 100) -> NDArray[np.float64]:
    """Déplacement d'une image par rapport à une référence, par corrélation croisée.

    Parameters
    ----------
    ref : ndarray, shape (ny, nx)
        Image de référence.
    img : ndarray, shape (ny, nx)
        Image déplacée.
    upsample : int, default 100
        Facteur de sur-échantillonnage du pic de corrélation (précision
        ``1/upsample`` pixel).

    Returns
    -------
    ndarray, shape (2,)
        Déplacement ``(dy, dx)`` de ``img`` par rapport à ``ref``, en pixels.
    """
    product = np.fft.fft2(ref) * np.fft.fft2(img).conj()
    corr = np.abs(np.fft.ifft2(product))
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape), dtype=float)
    shape = np.array(corr.shape)
    peak[peak > shape // 2] -= shape[peak > shape // 2]
    peak = np.round(peak * upsample) / upsample
    size = int(np.ceil(upsample * 1.5))
    centre = np.trunc(size / 2.0)
    corr = _upsampled_dft(product.conj(), size, upsample, centre - peak * upsample).conj()
    fine = np.array(np.unravel_index(np.argmax(np.abs(corr)), corr.shape), dtype=float)
    shift = peak + (fine - centre) / upsample
    return -shift  # décalage de recalage de img sur ref -> déplacement de img


def _denoise(frames: NDArray[np.float64], sigma: ArrayLike, nsig: float) -> NDArray[np.float64]:
    """Seuillage dur à ``nsig`` écarts types : garde l'étoile, retire le bruit."""
    return np.array([hard_threshold(f, nsig * s) for f, s in zip(frames, np.asarray(sigma))])


def fourier_shift(img: NDArray[np.float64], shift: ArrayLike) -> NDArray[np.float64]:
    """Déplace une image par multiplication de phase dans le plan de Fourier.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image (supposée périodique et à bande limitée).
    shift : array_like, shape (2,)
        Déplacement ``(dy, dx)`` en pixels.

    Returns
    -------
    ndarray, shape (ny, nx)
        Image déplacée.
    """
    dy, dx = np.asarray(shift, dtype=float)
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.rfftfreq(img.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.fft.irfft2(np.fft.rfft2(img) * phase, s=img.shape)


def fit_frame(
    frame: NDArray[np.float64],
    model: NDArray[np.float64],
    upfact: int,
    sigma: float,
    init_shift: ArrayLike,
    init_flux: float = 1.0,
    window: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Ajuste décalage et flux d'une image BR sur un modèle HR.

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        Image BR.
    model : ndarray, shape (ny * upfact, nx * upfact)
        Modèle HR aligné sur l'image de référence.
    upfact : int
        Facteur de sur-échantillonnage.
    sigma : float
        Niveau de bruit de l'image.
    init_shift : array_like, shape (2,)
        Décalage initial ``(dy, dx)``, en pixels BR.
    init_flux : float, default 1.0
        Flux initial.
    window : ndarray of bool, shape (ny, nx), optional
        Pixels pris en compte (pixels valides proches de l'étoile) ; tous par
        défaut.

    Returns
    -------
    shift : ndarray, shape (2,)
        Décalage ajusté, en pixels BR.
    flux : float
        Flux ajusté.
    """
    sel = np.ones(frame.shape, bool) if window is None else window
    init = np.asarray(init_shift, dtype=float)

    def resid(p: NDArray[np.float64]) -> NDArray[np.float64]:
        pred = p[2] * fourier_shift(model, (upfact * p[0], upfact * p[1]))[::upfact, ::upfact]
        return ((frame - pred)[sel] / sigma).ravel()

    res = least_squares(resid, [init[0], init[1], init_flux], x_scale=[0.1, 0.1, 0.1])
    return res.x[:2], float(res.x[2])


def estimate_shifts(
    frames: NDArray[np.float64],
    sigma: ArrayLike,
    psf_sigma: float,
    method: str = "centroid_wide",
    ref: int = 0,
    nsig: float = 4.0,
    wide_factor: float = 2.5,
    upsample: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estime les décalages des images par rapport à l'image de référence.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR (pixels invalides déjà interpolés).
    sigma : array_like, shape (n_frames,)
        Niveau de bruit de chaque image.
    psf_sigma : float
        Largeur de la PSF, en pixels BR.
    method : {"centroid_wide", "centroid", "xcorr"}, default "centroid_wide"
        Estimateur.
    ref : int, default 0
        Image de référence.
    nsig : float, default 4.0
        Seuil de bruit, en écarts types.
    wide_factor : float, default 2.5
        Largeur de fenêtre de ``centroid_wide``, en unités de ``psf_sigma``.
    upsample : int, default 100
        Sur-échantillonnage du pic de corrélation de ``xcorr``.

    Returns
    -------
    shifts : ndarray, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels BR ; celui de la référence est nul.
    centroids : ndarray, shape (n_frames, 2)
        Centroïdes comptés à partir de 1, utilisés pour l'ouverture de flux.

    Raises
    ------
    ValueError
        Si la méthode n'est pas reconnue.
    """
    cents = estimate_centroids(frames, sigma, psf_sigma, nsig)
    if method == "centroid":
        return cents - cents[ref], cents
    if method == "centroid_wide":
        wide = estimate_centroids(frames, sigma, wide_factor * psf_sigma, nsig)
        return wide - wide[ref], wide
    if method == "xcorr":
        clean = _denoise(frames, sigma, nsig)
        shifts = np.array([xcorr_shift(clean[ref], c, upsample) for c in clean])
        return shifts - shifts[ref], cents
    raise ValueError(f"méthode de décalage inconnue {method!r} ; choisir parmi {METHODS}")


def refine_on_model(
    frames: NDArray[np.float64],
    sigma: ArrayLike,
    model: NDArray[np.float64],
    upfact: int,
    shifts: NDArray[np.float64],
    flux: NDArray[np.float64],
    ref: int = 0,
    valid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Réajuste décalages et flux sur un modèle HR donné (une solution SPRITE).

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    sigma : array_like, shape (n_frames,)
        Niveau de bruit de chaque image.
    model : ndarray, shape (ny * upfact, nx * upfact)
        Modèle HR.
    upfact : int
        Facteur de sur-échantillonnage.
    shifts : ndarray, shape (n_frames, 2)
        Décalages initiaux, en pixels BR.
    flux : ndarray, shape (n_frames,)
        Flux initiaux.
    ref : int, default 0
        Image de référence.
    valid : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels valides ; les autres sont ignorés.

    Returns
    -------
    shifts : ndarray, shape (n_frames, 2)
        Décalages relatifs à la référence, en pixels BR.
    flux : ndarray, shape (n_frames,)
        Flux relatifs à la référence.
    """
    c = weighted_centroid(frames[ref], 3.0) - 1
    yy, xx = np.indices(frames.shape[1:])
    window = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= 12 ** 2
    masks = [window] * len(frames) if valid is None else [window & v for v in valid]
    fitted = [fit_frame(f, model, upfact, s, sh, fl, w)
              for f, s, sh, fl, w in zip(frames, np.asarray(sigma), shifts, flux, masks)]
    new_shifts = np.array([p for p, _ in fitted])
    new_flux = np.array([a for _, a in fitted])
    return new_shifts - new_shifts[ref], new_flux / new_flux[ref]
