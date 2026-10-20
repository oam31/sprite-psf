"""Estimation de la taille de la PSF (``Sprite::lr_psf_size``).

Le C++ co-additionne les images avec le shift-and-add d'iSAP/eclipse
(``coadd_frames`` : recalage par somme des carrés des différences sur l'image
0, rééchantillonnage par noyau tanh, médiane), débruite le résultat par support
multirésolution, puis ajuste une gaussienne. La largeur obtenue sert de fenêtre
aux centroïdes et fixe le rayon de l'ouverture de flux.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .estimation import gaussian_fit
from .first_guess import mr_support_filter
from .wavelets import Transform

_TABSPERPIX = 1000
_KERNEL_WIDTH = 2.0
_TANH_STEEPNESS = 5.0


def _tanh_kernel(steep: float = _TANH_STEEPNESS) -> NDArray[np.float64]:
    """Noyau d'interpolation tanh tabulé (``generate_tanh_kernel``)."""
    width = _TABSPERPIX / 2.0
    npts = 32768
    i = np.arange(npts)
    ind = np.where(i < npts // 2, i, i - npts) * 2.0 * width / npts
    hk = ((np.tanh(steep * (ind + 0.5)) + 1) / 2) * ((np.tanh(steep * (-ind + 0.5)) + 1) / 2)
    samples = 1 + int(_TABSPERPIX * _KERNEL_WIDTH)
    return 2.0 * width * np.fft.ifft(hk).real[:samples]


def _ssd_offset(
    pattern: NDArray[np.float64],
    frame: NDArray[np.float64],
    center: tuple[int, int],
    max_dist: int = 4,
    half: int = 10,
) -> tuple[float, float]:
    """Décalage sous-pixel minimisant la somme des carrés des différences (``private_correlation``)."""
    cy, cx = center
    ref = frame[cy - half:cy + half + 1, cx - half:cx + half + 1]
    n = 2 * max_dist + 1
    dist = np.empty((n, n))
    for l in range(-max_dist, max_dist + 1):
        for k in range(-max_dist, max_dist + 1):
            win = pattern[cy + l - half:cy + l + half + 1, cx + k - half:cx + k + half + 1]
            dist[l + max_dist, k + max_dist] = ((win - ref) ** 2).sum()
    dist /= (2 * half + 1) ** 2
    l_min, k_min = np.unravel_index(np.argmin(dist), dist.shape)

    def apodisation(d1: float, d2: float, d3: float) -> float:
        if d1 == d2:
            return -0.5
        if d2 == d3:
            return 0.5
        if d1 == d3:
            return 0.0
        return float(np.clip(0.5 * (d1 - d3) / (d1 - 2 * d2 + d3), -0.5, 0.5))

    inc_x = 0.0 if k_min in (0, n - 1) else apodisation(*dist[l_min, k_min - 1:k_min + 2])
    inc_y = 0.0 if l_min in (0, n - 1) else apodisation(*dist[l_min - 1:l_min + 2, k_min])
    return (l_min - max_dist) + inc_y, (k_min - max_dist) + inc_x


def _warp(img: NDArray[np.float64], dy: float, dx: float, zoom: float, kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Décalage et zoom par le noyau tabulé 4 × 4 (``warp_image``)."""
    ly, lx = img.shape
    out = np.zeros((int(zoom * ly), int(zoom * lx)))
    for j in range(out.shape[0]):
        y = j / zoom - dy
        py = int(y)
        if py < 1 or py > ly - 3:
            continue
        ty = int((y - py) * _TABSPERPIX)
        ry = kernel[[_TABSPERPIX + ty, ty, _TABSPERPIX - ty, 2 * _TABSPERPIX - ty]]
        for i in range(out.shape[1]):
            x = i / zoom - dx
            px = int(x)
            if px < 1 or px > lx - 3:
                continue
            tx = int((x - px) * _TABSPERPIX)
            rx = kernel[[_TABSPERPIX + tx, tx, _TABSPERPIX - tx, 2 * _TABSPERPIX - tx]]
            patch = img[py - 1:py + 3, px - 1:px + 3]
            out[j, i] = ry @ patch @ rx / (rx.sum() * ry.sum())
    return out


def coadd(frames: NDArray[np.float64], zoom: int = 2, max_dist: int = 4, surface: int = 10) -> NDArray[np.float64]:
    """Images recalées, zoomées et combinées par médiane (valeurs par défaut de ``coadd_frames``).

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    zoom : int, default 2
        Facteur de zoom.
    max_dist : int, default 4
        Demi-largeur de la zone de recherche du décalage, en pixels.
    surface : int, default 10
        Demi-largeur de la fenêtre de comparaison, en pixels.

    Returns
    -------
    ndarray, shape (ny * zoom, nx * zoom)
        Médiane (inférieure) des images recalées sur l'image 0.
    """
    nz, ny, nx = frames.shape
    center = (ny // 2, nx // 2)
    # Comme le C++ : fenêtre réduite si elle ne tient pas dans l'image.
    room = min(center[0], center[1], ny - center[0] - 1, nx - center[1] - 1) - max_dist
    max_dist = max(1, min(max_dist, min(center) // 4))
    surface = max(1, min(surface, room - 1, min(center) - max_dist - 1))
    kernel = _tanh_kernel()
    warped = []
    for f in frames:
        dy, dx = _ssd_offset(frames[0], f, center, max_dist, surface)
        warped.append(_warp(f, dy, dx, zoom, kernel))
    stack = np.array(warped)
    k = nz // 2 if nz % 2 else nz // 2 - 1
    return np.partition(stack, k, axis=0)[k]


def psf_sigma(frames: NDArray[np.float64], upfact: int, transform: Transform, nsig: float = 5.0) -> float:
    """Largeur gaussienne de la PSF co-additionnée et débruitée.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    upfact : int
        Facteur de zoom de la co-addition.
    transform : Starlet or Undecimated79
        Transformée du débruitage (7/9 non décimée dans SPRITE).
    nsig : float, default 5.0
        Seuil du débruitage.

    Returns
    -------
    float
        Moyenne des écarts types ajustés selon les deux axes, en pixels BR.
    """
    model = mr_support_filter(coadd(frames, upfact), transform, nsig)
    sy, sx = gaussian_fit(model)[:2]
    return (abs(sy) + abs(sx)) / (2 * upfact)
