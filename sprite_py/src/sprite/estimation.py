"""Paramètres d'attache aux données estimés sur les images BR (section 3.4.1 du papier).

Conventions : les images sont des tableaux ``(n_frames, ny, nx)`` dans
l'orientation naturelle de numpy. Les centroïdes et les décalages sont des
tableaux ``(n_frames, 2)`` ordonnés ``(y, x)`` ; les centroïdes sont comptés à
partir de 1, comme dans le C++.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.optimize import least_squares

from .wavelets import Undecimated79, mad_sigma


def noise_sigma(frames: NDArray[np.float64]) -> NDArray[np.float64]:
    """Niveau de bruit de chaque image : MAD de la bande de détails la plus fine.

    Reproduit ``noise_est`` (transformée 7/9 non décimée à 2 échelles, normée
    en L2). Le C++ travaille sur les images transposées et prend sa première
    bande (horizontale), qui est la bande verticale (indice 1) dans
    l'orientation naturelle.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.

    Returns
    -------
    ndarray, shape (n_frames,)
        Écart type du bruit de chaque image.
    """
    wt = Undecimated79(2, norm="l2")
    return np.array([mad_sigma(wt.transform(f)[1]) for f in frames])


def weighted_centroid(img: NDArray[np.float64], sigma: float, n_iter: int = 10) -> NDArray[np.float64]:
    """Centroïde à pondération gaussienne itérative (Baker et Moallem 2007).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image.
    sigma : float
        Écart type de la fenêtre gaussienne, en pixels.
    n_iter : int, default 10
        Nombre d'itérations ; la première est non pondérée.

    Returns
    -------
    ndarray, shape (2,)
        Centroïde ``(y, x)``, compté à partir de 1.
    """
    yy, xx = np.indices(img.shape) + 1.0
    w = np.ones_like(img)
    cy = cx = 0.0
    for k in range(n_iter):
        if k > 0:
            w = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        wimg = img * w
        tot = wimg.sum()
        cy, cx = (wimg * yy).sum() / tot, (wimg * xx).sum() / tot
    return np.array([cy, cx])


def hard_threshold(x: NDArray[np.float64], thresh: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Seuillage dur : met à zéro les valeurs de module inférieur au seuil.

    Parameters
    ----------
    x : ndarray
        Valeurs.
    thresh : float or ndarray
        Seuil, scalaire ou de même forme que ``x``.

    Returns
    -------
    ndarray
        Valeurs seuillées.
    """
    return np.where(np.abs(x) < thresh, 0.0, x)


def estimate_centroids(
    frames: NDArray[np.float64], sigma: ArrayLike, psf_sigma: float, nsig: float = 4.0
) -> NDArray[np.float64]:
    """Centroïdes des images seuillées (éq. 24 du papier, ``shift_est``).

    Chaque image est seuillée à ``min(nsig sigma, (floor(max / sigma) - 1) sigma)``
    avant le calcul du centroïde pondéré.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    sigma : array_like, shape (n_frames,)
        Niveau de bruit de chaque image.
    psf_sigma : float
        Largeur de la fenêtre de pondération, en pixels BR.
    nsig : float, default 4.0
        Seuil en nombre d'écarts types.

    Returns
    -------
    ndarray, shape (n_frames, 2)
        Centroïdes ``(y, x)``, comptés à partir de 1.
    """
    cents = []
    for img, s in zip(frames, np.asarray(sigma)):
        psnr = np.floor(img.max() / s)
        thresh = (psnr - 1) * s if psnr < nsig else nsig * s
        cents.append(weighted_centroid(hard_threshold(img, thresh), psf_sigma))
    return np.array(cents)


def estimate_flux(
    frames: NDArray[np.float64],
    centroids: NDArray[np.float64],
    radius: float,
    ref: int = 0,
    cpp_compat: bool = False,
) -> NDArray[np.float64]:
    """Flux relatifs dans un disque autour de chaque centroïde (``flux_est``).

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    centroids : ndarray, shape (n_frames, 2)
        Centroïdes comptés à partir de 1.
    radius : float
        Rayon de l'ouverture, en pixels BR (``4 * psf_sigma`` dans SPRITE).
    ref : int, default 0
        Image de référence, de flux 1.
    cpp_compat : bool, default False
        Le C++ compare des indices comptés à partir de 0 aux centroïdes
        comptés à partir de 1, ce qui décale l'ouverture d'un pixel sur chaque
        axe ; ``True`` reproduit ce défaut.

    Returns
    -------
    ndarray, shape (n_frames,)
        Flux divisés par celui de l'image de référence.
    """
    yy, xx = np.indices(frames.shape[1:])
    if not cpp_compat:
        yy, xx = yy + 1, xx + 1
    flux = np.empty(len(frames))
    for k, (img, c) in enumerate(zip(frames, centroids)):
        inside = (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= radius ** 2
        flux[k] = img[inside].sum()
    return flux / flux[ref]


def fit_flux(
    frames: NDArray[np.float64],
    model: NDArray[np.float64],
    hr_shifts: NDArray[np.float64],
    upfact: int,
    valid: NDArray[np.bool_] | None = None,
    ref: int = 0,
    lanczos_radius: int = 10,
) -> NDArray[np.float64]:
    """Flux relatifs par ajustement linéaire sur un modèle HR (photométrie par PSF).

    Pour chaque image, ``m_k = D H_k model`` est l'image prédite à flux 1 ; le
    flux optimal au sens des moindres carrés est
    ``f_k = <y_k, m_k> / <m_k, m_k>`` sur les pixels valides.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    model : ndarray, shape (ny * upfact, nx * upfact)
        Modèle HR aligné sur l'image de référence (première estimation ou
        solution).
    hr_shifts : ndarray, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels HR.
    upfact : int
        Facteur de sur-échantillonnage.
    valid : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels valides ; tous par défaut.
    ref : int, default 0
        Image de référence, de flux 1.
    lanczos_radius : int, default 10
        Rayon des noyaux de recalage.

    Returns
    -------
    ndarray, shape (n_frames,)
        Flux divisés par celui de l'image de référence.
    """
    from .operators import SpriteOperator

    n = len(frames)
    op = SpriteOperator(hr_shifts, np.ones(n), np.ones(n), frames.shape[1:], upfact, lanczos_radius)
    pred = op.op(model)
    sel = np.ones(frames.shape, bool) if valid is None else valid
    flux = np.array([(f[v] * m[v]).sum() / (m[v] ** 2).sum() for f, m, v in zip(frames, pred, sel)])
    return flux / flux[ref]


def _gauss(amp: float, sy: float, sx: float, cy: float, cx: float, shape: tuple[int, int]) -> NDArray[np.float64]:
    yy, xx = np.indices(shape)
    return amp * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)


def gaussian_fit(img: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    """Ajustement par moindres carrés d'une gaussienne 2D alignée sur les axes.

    Le C++ utilise son propre schéma de Newton alterné (``gauss2d_fit_2``) ;
    un ajustement standard suffit ici, seule la largeur étant utilisée.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image contenant une source.

    Returns
    -------
    tuple of float
        ``(sy, sx, cy, cx, amp)`` : écarts types, centre (compté à partir de 0)
        et amplitude.
    """
    smooth = ndimage.uniform_filter(img, 3, mode="constant")
    cy, cx = np.unravel_index(np.argmax(smooth), img.shape)
    amp = img[cy, cx]
    level = amp / np.e
    sy = max(1.0, float(np.count_nonzero(img[:, cx] > level)) / 2)
    sx = max(1.0, float(np.count_nonzero(img[cy, :] > level)) / 2)

    def resid(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return (_gauss(p[4], p[0], p[1], p[2], p[3], img.shape) - img).ravel()

    res = least_squares(resid, [sy, sx, cy, cx, amp])
    return tuple(float(v) for v in res.x)
