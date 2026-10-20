"""Pixels saturés ou défectueux.

Un pixel défectueux (chaud, mort, touché par un rayon cosmique, valeur non
finie) ne porte aucune information : il reçoit un poids nul dans l'attache aux
données. Un pixel saturé en porte une : la vraie valeur est au moins égale au
niveau de saturation. Il entre donc dans l'attache aux données par un terme
censuré, une pénalité quadratique appliquée seulement quand le modèle passe
sous la valeur mesurée (voir ``solver.SpriteGradient``). Ce terme est convexe
et dérivable : le forward-backward généralisé s'applique sans changement.

Les estimations préalables (bruit, taille de la PSF, décalages, flux, première
estimation) opèrent sur des images complètes : les pixels défectueux y sont
remplacés par une interpolation harmonique (:func:`inpaint`) ; les pixels
saturés gardent leur valeur écrêtée. L'interpolation n'entre jamais dans
l'attache aux données.

Mesuré sur données simulées : exclure les pixels saturés au lieu de les
censurer dégrade la reconstruction, car le pic n'est alors plus contraint.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


def build_masks(
    frames: NDArray[np.float64],
    saturation: float | None = None,
    bad_pixels: ArrayLike | None = None,
    dilate: int = 0,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Classe les pixels en valides, censurés (saturés) et exclus.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR brutes, avant soustraction du fond.
    saturation : float, optional
        Niveau de saturation, en unités des données brutes : tout pixel ``>=``
        à ce niveau est censuré.
    bad_pixels : array_like of bool, optional
        Pixels défectueux connus (``True`` = défectueux), de forme ``(ny, nx)``
        (carte commune à toutes les images) ou ``(n_frames, ny, nx)``.
    dilate : int, default 0
        Largeur, en pixels, de la couronne exclue autour des zones saturées ou
        défectueuses (débordement de charge, par exemple).

    Returns
    -------
    valid : ndarray of bool, shape (n_frames, ny, nx)
        Pixels pris en compte normalement.
    censored : ndarray of bool, shape (n_frames, ny, nx)
        Pixels saturés, pris en compte comme bornes inférieures. Les autres
        pixels (ni valides ni censurés) sont exclus.

    Raises
    ------
    ValueError
        Si la carte des pixels défectueux n'a pas une forme compatible.
    """
    bad = ~np.isfinite(frames)
    if bad_pixels is not None:
        known = np.asarray(bad_pixels, dtype=bool)
        if known.shape not in (frames.shape, frames.shape[1:]):
            raise ValueError(
                f"carte des pixels défectueux de forme {known.shape}, attendue "
                f"{frames.shape[1:]} ou {frames.shape}"
            )
        bad |= np.broadcast_to(known, frames.shape)
    saturated = np.zeros(frames.shape, bool)
    if saturation is not None:
        with np.errstate(invalid="ignore"):
            saturated = (frames >= saturation) & ~bad
    excluded = bad.copy()
    if dilate > 0:
        structure = np.zeros((1, 3, 3), bool)
        structure[0] = ndimage.generate_binary_structure(2, 2)
        ring = ndimage.binary_dilation(bad | saturated, structure=structure, iterations=dilate)
        excluded |= ring & ~saturated
    valid = ~(excluded | saturated)
    return valid, saturated


def build_mask(
    frames: NDArray[np.float64],
    saturation: float | None = None,
    bad_pixels: ArrayLike | None = None,
    dilate: int = 0,
) -> NDArray[np.bool_]:
    """Masque des pixels pleinement valides (ni saturés ni exclus).

    Raccourci de :func:`build_masks` ; mêmes paramètres.

    Returns
    -------
    ndarray of bool, shape (n_frames, ny, nx)
        ``True`` pour les pixels valides.
    """
    return build_masks(frames, saturation, bad_pixels, dilate)[0]


def inpaint(
    frames: NDArray[np.float64], valid: NDArray[np.bool_], max_iter: int = 100, n_relax: int = 200
) -> NDArray[np.float64]:
    """Remplace les pixels invalides par une interpolation harmonique de leurs voisins.

    Deux temps :

    1. remplissage de l'extérieur vers l'intérieur des zones invalides : à
       chaque passe, chaque pixel invalide ayant au moins un voisin connu
       (8-connexité) prend la moyenne de ces voisins ;
    2. relaxation : les pixels invalides prennent, ``n_relax`` fois, la moyenne
       de leurs 4 voisins. La limite est la solution de l'équation de Laplace
       à bords fixés, qui reproduit exactement une variation linéaire.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images.
    valid : ndarray of bool, shape (n_frames, ny, nx)
        Masque des pixels valides.
    max_iter : int, default 100
        Nombre maximal de passes de remplissage.
    n_relax : int, default 200
        Nombre de passes de relaxation.

    Returns
    -------
    ndarray, shape (n_frames, ny, nx)
        Copie des images, pixels invalides remplacés. Une image sans aucun
        pixel valide est renvoyée nulle.
    """
    out = np.where(valid, frames, 0.0).astype(float)
    known = valid.copy()
    kernel = np.ones((3, 3))
    cross = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for k in range(out.shape[0]):
        if not known[k].any():
            out[k] = 0.0
            continue
        for _ in range(max_iter):
            missing = ~known[k]
            if not missing.any():
                break
            total = ndimage.convolve(out[k] * known[k], kernel, mode="constant")
            count = ndimage.convolve(known[k].astype(float), kernel, mode="constant")
            fill = missing & (count > 0)
            out[k][fill] = total[fill] / count[fill]
            known[k] |= fill
        holes = ~valid[k]
        for _ in range(n_relax):
            avg = ndimage.convolve(out[k], cross, mode="nearest") / 4.0
            out[k][holes] = avg[holes]
    return out


def summary(valid: NDArray[np.bool_], censored: NDArray[np.bool_] | None = None) -> str:
    """Résumé lisible des masques, par image.

    Parameters
    ----------
    valid : ndarray of bool, shape (n_frames, ny, nx)
        Pixels valides.
    censored : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels saturés (censurés).

    Returns
    -------
    str
        Une ligne, par exemple ``"pixels saturés : 1, 0, 2, 1 ; exclus : 3, 0, 5, 1"``.
    """
    sat = np.zeros_like(valid) if censored is None else censored
    excluded = ~valid & ~sat
    fmt = lambda m: ", ".join(str(int(c)) for c in m.reshape(m.shape[0], -1).sum(axis=1))
    return f"pixels saturés : {fmt(sat)} ; exclus : {fmt(excluded)}"
