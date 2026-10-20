"""Première estimation x0 (section 3.4.2 du papier) : shift-and-add de Lanczos, puis débruitage.

Reproduit ``Sprite::mod_gen_2`` avec ses valeurs par défaut : médiane des
images sur-échantillonnées par un noyau de Lanczos de rayon 5, puis une passe
de filtrage par support multirésolution (``mr_support_filt``) avec la
transformée 7/9 non décimée, l'échelle grossière supprimée et une contrainte de
positivité.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import hard_threshold
from .operators import lanczos_1d
from .wavelets import Transform, mad_sigma


def _convolve_centered(signal: NDArray[np.float64], taps: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Convolution iSAP ``convol1D_cent`` le long de ``axis`` (bords nuls).

    ``out[i] = sum_k in[i - k] h[rad - k]`` pour ``k`` dans
    ``[max(-rad, i - (n - 1)), min(i, rad))`` : la borne supérieure est exclue,
    si bien que le dernier coefficient disparaît pour ``i >= rad`` et le
    coefficient central pour ``i = 0`` (comportement du C++, conservé).
    """
    sig = np.moveaxis(signal, axis, -1)
    n = sig.shape[-1]
    rad = (taps.size - 1) // 2
    out = np.zeros_like(sig)
    for i in range(n):
        ks = np.arange(max(-rad, i - (n - 1)), min(i, rad))
        if ks.size:
            out[..., i] = sig[..., i - ks] @ taps[rad - ks]
    return np.moveaxis(out, -1, axis)


def lanczos_upsample(
    img: NDArray[np.float64], upfact: int, shift: ArrayLike, radius: int = 5
) -> NDArray[np.float64]:
    """Sur-échantillonne une image par un interpolateur de Lanczos décalé.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image BR.
    upfact : int
        Facteur de sur-échantillonnage.
    shift : array_like, shape (2,)
        Décalage ``(u_lignes, u_colonnes)`` : le pixel HR ``J`` d'un axe
        échantillonne le signal BR en ``J / upfact + u``.
    radius : int, default 5
        Rayon du noyau (valeur du C++).

    Returns
    -------
    ndarray, shape (ny * upfact, nx * upfact)
        Image HR.
    """
    u_rows, u_cols = np.asarray(shift, dtype=float)
    ny, nx = img.shape
    tmp = np.empty((ny, nx * upfact))
    for p in range(upfact):
        taps = lanczos_1d(u_cols + p / upfact, radius)
        tmp[:, p::upfact] = _convolve_centered(img, taps, axis=1)
    out = np.empty((ny * upfact, nx * upfact))
    for p in range(upfact):
        taps = lanczos_1d(u_rows + p / upfact, radius)
        out[p::upfact, :] = _convolve_centered(tmp, taps, axis=0)
    return out


def lanczos_stack(
    frames: NDArray[np.float64],
    upfact: int,
    lr_shifts: ArrayLike,
    radius: int = 5,
    cpp_compat: bool = False,
) -> NDArray[np.float64]:
    """Médiane pixel à pixel des images recalées et sur-échantillonnées.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR.
    upfact : int
        Facteur de sur-échantillonnage.
    lr_shifts : array_like, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels BR : l'image ``k`` est la référence
        déplacée de ``+décalage`` ; elle est rééchantillonnée en ``+décalage``
        pour la recaler.
    radius : int, default 5
        Rayon du noyau de Lanczos.
    cpp_compat : bool, default False
        Le C++ (``lanczos_stacking``) rééchantillonne au décalage opposé et sur
        les axes permutés ; ``True`` reproduit ce défaut.

    Returns
    -------
    ndarray, shape (ny * upfact, nx * upfact)
        Médiane (inférieure pour un nombre pair d'images, comme iSAP).
    """
    ups = []
    for img, (dy, dx) in zip(frames, np.asarray(lr_shifts, dtype=float)):
        shift = (-dx, -dy) if cpp_compat else (dy, dx)
        ups.append(lanczos_upsample(img, upfact, shift, radius))
    stack = np.array(ups)
    n = len(frames)
    k = n // 2 if n % 2 else n // 2 - 1
    return np.partition(stack, k, axis=0)[k]


def _band_support(band: NDArray[np.float64], nsig: float = 5.0, n_iter: int = 5) -> NDArray[np.bool_]:
    """Coefficients significatifs d'une bande (``mad_res_noise_est`` puis ``check_ineq``)."""
    est = np.zeros_like(band)
    res = band
    for _ in range(n_iter):
        res = band - est
        est = est + res
        sig = mad_sigma(res)
        est = hard_threshold(est, nsig * sig)
    sig = mad_sigma(res)
    return band > nsig * sig


def mr_support_filter(img: NDArray[np.float64], transform: Transform, nsig: float = 5.0) -> NDArray[np.float64]:
    """Filtrage en une passe par support multirésolution, échelle grossière supprimée, x >= 0.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image à débruiter.
    transform : Starlet or Undecimated79
        Transformée utilisée (7/9 non décimée dans SPRITE).
    nsig : float, default 5.0
        Seuil de significativité, en écarts types (positifs seulement).

    Returns
    -------
    ndarray, shape (ny, nx)
        Image reconstruite à partir des seuls coefficients significatifs,
        rendue positive.
    """
    coeffs = transform.transform(img)
    for b in range(coeffs.shape[0] - 1):
        coeffs[b] *= _band_support(coeffs[b], nsig)
    coeffs[-1] = 0.0
    return np.maximum(transform.recons(coeffs), 0.0)


def first_guess(
    frames: NDArray[np.float64],
    upfact: int,
    lr_shifts: ArrayLike,
    transform: Transform,
    nsig: float = 5.0,
    cpp_compat: bool = False,
) -> NDArray[np.float64]:
    """Première estimation ``mod_gen_2`` : pile de Lanczos puis support multirésolution.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR (pixels invalides déjà interpolés).
    upfact : int
        Facteur de sur-échantillonnage.
    lr_shifts : array_like, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels BR.
    transform : Starlet or Undecimated79
        Transformée du débruitage.
    nsig : float, default 5.0
        Seuil du débruitage.
    cpp_compat : bool, default False
        Reproduit le défaut de recalage du C++.

    Returns
    -------
    ndarray, shape (ny * upfact, nx * upfact)
        Première estimation, positive.
    """
    stack = lanczos_stack(frames, upfact, lr_shifts, cpp_compat=cpp_compat)
    return mr_support_filter(stack, transform, nsig)
