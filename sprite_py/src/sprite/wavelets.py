"""Dictionnaires d'ondelettes non décimées utilisés par SPRITE.

Les deux transformées reproduisent les implémentations iSAP livrées avec
SPRITE v1.0 :

* :class:`Starlet` : transformée isotrope « à trous » B3-spline (iSAP
  ``TO_PAVE_BSPLINE``, ``run_sprite -t 2``), bord constant ;
* :class:`Undecimated79` : transformée bi-orthogonale 7/9 non décimée, à trois
  bandes directionnelles par échelle (iSAP ``TO_UNDECIMATED_MALLAT``,
  ``-t 24``), bord miroir.

Les coefficients sont rangés dans un tableau ``(n_bandes, ny, nx)`` dont la
dernière bande est l'échelle grossière. ``recons`` est l'opérateur de
reconstruction d'iSAP (un inverse, pas un adjoint), comme dans le solveur C++.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Filtres 7/9 d'Antonini tels que tabulés dans iSAP (Filter.cc). Pour
# F_MALLAT_7_9, le filtre d'analyse est celui à 7 coefficients et le filtre de
# synthèse celui à 9 coefficients.
_ANTONINI_7 = np.array([
    -6.453888262893856e-02, -4.068941760955867e-02, 4.180922732222124e-01,
    7.884856164056651e-01, 4.180922732222124e-01, -4.068941760955867e-02,
    -6.453888262893856e-02,
])
_ANTONINI_9 = np.array([
    3.782845550699535e-02, -2.384946501937986e-02, -1.106244044184226e-01,
    3.774028556126536e-01, 8.526986790094022e-01, 3.774028556126537e-01,
    -1.106244044184226e-01, -2.384946501937986e-02, 3.782845550699535e-02,
])


def default_n_scales(n: int) -> int:
    """Nombre d'échelles par défaut d'iSAP.

    Parameters
    ----------
    n : int
        Plus petite dimension de l'image, en pixels.

    Returns
    -------
    int
        ``floor(log2(n)) - 1``.
    """
    return int(np.floor(np.log(n) / np.log(2))) - 1


def isap_median(values: ArrayLike) -> float:
    """Médiane au sens d'iSAP (``get_median``).

    Parameters
    ----------
    values : array_like
        Valeurs, aplaties.

    Returns
    -------
    float
        Médiane ; pour un effectif pair, l'élément inférieur des deux
        éléments centraux.
    """
    v = np.ravel(values)
    n = v.size
    k = n // 2 if n % 2 else n // 2 - 1
    return float(np.partition(v, k)[k])


def mad_sigma(values: ArrayLike) -> float:
    """Écart type d'un bruit gaussien estimé par le MAD (iSAP ``get_sigma_mad``).

    Parameters
    ----------
    values : array_like
        Échantillon dominé par le bruit.

    Returns
    -------
    float
        ``median(|v - median(v)|) / 0.6745``, avec la médiane d'iSAP.
    """
    med = isap_median(values)
    return isap_median(np.abs(np.ravel(values) - med)) / 0.6745


def _mirror(idx: NDArray[np.int_], n: int) -> NDArray[np.int_]:
    """Réflexion sans répétition du bord (iSAP ``test_index_mirror``)."""
    idx = np.where(idx < 0, np.minimum(-idx, n - 1), idx)
    return np.where(idx >= n, np.maximum(2 * (n - 1) - idx, 0), idx)


def _clamp(idx: NDArray[np.int_], n: int) -> NDArray[np.int_]:
    """Prolongement par la valeur du bord (iSAP ``test_index_cont``)."""
    return np.clip(idx, 0, n - 1)


_BORDERS = {"_mirror": _mirror, "_clamp": _clamp}


def _filter_axis(
    x: NDArray[np.float64],
    taps: NDArray[np.float64],
    offsets: NDArray[np.int_],
    axis: int,
    border: Callable[[NDArray[np.int_], int], NDArray[np.int_]],
) -> NDArray[np.float64]:
    """Calcule ``sum_p taps[p] * x[border(i + offsets[p])]`` le long de ``axis``.

    Le filtrage est appliqué comme un produit par la matrice ``A`` de
    :func:`_filter_matrix`, mise en cache.
    """
    n = x.shape[axis]
    mat = _filter_matrix(n, tuple(float(t) for t in taps), tuple(int(o) for o in offsets), border.__name__)
    return np.moveaxis(np.tensordot(mat, np.moveaxis(x, axis, 0), axes=(1, 0)), 0, axis)


@lru_cache(maxsize=256)
def _filter_matrix(
    n: int, taps: tuple[float, ...], offsets: tuple[int, ...], border_name: str
) -> NDArray[np.float64]:
    """Matrice ``A`` (n × n) du filtrage 1D : ``out = A @ x`` le long d'un axe."""
    border = _BORDERS[border_name]
    base = np.arange(n)
    mat = np.zeros((n, n))
    for tap, off in zip(taps, offsets):
        np.add.at(mat, (base, border(base + off, n)), tap)
    return mat


def _filter_axis_adjoint(
    y: NDArray[np.float64],
    taps: NDArray[np.float64],
    offsets: NDArray[np.int_],
    axis: int,
    border: Callable[[NDArray[np.int_], int], NDArray[np.int_]],
) -> NDArray[np.float64]:
    """Adjoint exact de :func:`_filter_axis`, bords compris.

    Le filtrage le long d'un axe est la matrice ``A`` de :func:`_filter_matrix` ;
    son adjoint est ``A^T``, appliquée par produit matriciel (matrice en cache).
    """
    n = y.shape[axis]
    mat = _filter_matrix(n, tuple(float(t) for t in taps), tuple(int(o) for o in offsets), border.__name__)
    return np.moveaxis(np.tensordot(mat.T, np.moveaxis(y, axis, 0), axes=(1, 0)), 0, axis)


class Starlet:
    """Transformée en ondelettes isotrope non décimée B3-spline (starlet).

    Parameters
    ----------
    n_scales : int
        Nombre d'échelles, échelle grossière comprise.
    """

    _B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

    def __init__(self, n_scales: int) -> None:
        self.n_scales = n_scales

    @property
    def n_bands(self) -> int:
        """Nombre de bandes, égal au nombre d'échelles."""
        return self.n_scales

    def _smooth(self, img: NDArray[np.float64], scale: int) -> NDArray[np.float64]:
        step = 2 ** scale
        offsets = np.arange(-2, 3) * step
        tmp = _filter_axis(img, self._B3, offsets, axis=1, border=_clamp)
        return _filter_axis(tmp, self._B3, offsets, axis=0, border=_clamp)

    def transform(self, img: ArrayLike) -> NDArray[np.float64]:
        """Décompose une image.

        Parameters
        ----------
        img : array_like, shape (ny, nx)
            Image à décomposer.

        Returns
        -------
        ndarray, shape (n_scales, ny, nx)
            Bandes de détails de la plus fine à la plus grossière, puis
            l'échelle grossière.
        """
        coeffs = np.empty((self.n_bands,) + np.shape(img))
        current = np.asarray(img, dtype=float)
        for s in range(self.n_scales - 1):
            smooth = self._smooth(current, s)
            coeffs[s] = current - smooth
            current = smooth
        coeffs[-1] = current
        return coeffs

    def recons(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruit l'image par somme des bandes.

        Parameters
        ----------
        coeffs : ndarray, shape (n_scales, ny, nx)
            Coefficients renvoyés par :meth:`transform`.

        Returns
        -------
        ndarray, shape (ny, nx)
            Image reconstruite.
        """
        return coeffs.sum(axis=0)

    def adjoint(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applique l'adjoint exact de :meth:`transform`.

        Diffère de :meth:`recons` : la starlet est un repère redondant, son
        inverse (somme des bandes) n'est pas son adjoint.

        Parameters
        ----------
        coeffs : ndarray, shape (n_scales, ny, nx)
            Coefficients.

        Returns
        -------
        ndarray, shape (ny, nx)
            ``Phi^T coeffs``.
        """
        g = coeffs[-1].copy()
        for s in range(self.n_scales - 2, -1, -1):
            step = 2 ** s
            offsets = np.arange(-2, 3) * step
            w = coeffs[s]
            # c_{s+1} = S c_s et w_s = c_s - c_{s+1}, donc g_s = w_s + S^T (g_{s+1} - w_s)
            t = _filter_axis_adjoint(g - w, self._B3, offsets, axis=0, border=_clamp)
            t = _filter_axis_adjoint(t, self._B3, offsets, axis=1, border=_clamp)
            g = w + t
        return g


class _SubBandFilter:
    """Banc de filtres 1D à deux canaux non décimé (iSAP ``SubBandFilter``).

    Parameters
    ----------
    norm : {"l1", "l2"}
        Normalisation des filtres.
    """

    def __init__(self, norm: str) -> None:
        # Les filtres tabulés sont normalisés en L2 (somme = sqrt(2)).
        corr = 1.0 / np.sqrt(2.0) if norm == "l1" else 1.0
        self.norm_coef = 2.0 if norm == "l1" else 1.0
        self.h0 = _ANTONINI_7 * corr
        self.h1 = _ANTONINI_9 * corr
        self.g1 = self._complement(self.h0)
        self.g0 = self._complement(self.h1)

    @staticmethod
    def _complement(f: NDArray[np.float64]) -> NDArray[np.float64]:
        n = f.size
        first_pix = -(n // 2)
        sign = 1.0 if first_pix % 2 == 0 else -1.0
        out = np.empty(n)
        for i in range(n):
            out[n - i - 1] = sign * f[i]
            sign = -sign
        return out

    @staticmethod
    def _start(f: NDArray[np.float64]) -> int:
        return -(f.size // 2)

    def analysis(
        self, x: NDArray[np.float64], axis: int, step: int
    ) -> list[NDArray[np.float64]]:
        """Sorties basse fréquence et détails (``convol_filter``, coefficients inversés)."""
        out = []
        for f in (self.h0, self.g0):
            offsets = (np.arange(f.size) + self._start(f)) * step
            out.append(_filter_axis(x, f[::-1], offsets, axis, _mirror))
        return out

    def analysis_adjoint(
        self, low: NDArray[np.float64], det: NDArray[np.float64], axis: int, step: int
    ) -> NDArray[np.float64]:
        """Adjoint exact de :meth:`analysis`."""
        acc: NDArray[np.float64] | float = 0.0
        for f, x in ((self.h0, low), (self.g0, det)):
            offsets = (np.arange(f.size) + self._start(f)) * step
            acc = acc + _filter_axis_adjoint(x, f[::-1], offsets, axis, _mirror)
        return acc

    def synthesis(
        self, low: NDArray[np.float64], det: NDArray[np.float64], axis: int, step: int
    ) -> NDArray[np.float64]:
        """Inverse de :meth:`analysis` (``rec_convol_filter``)."""
        acc: NDArray[np.float64] | float = 0.0
        for f, x in ((self.h1, low), (self.g1, det)):
            offsets = (np.arange(f.size) + self._start(f)) * step
            acc = acc + _filter_axis(x, f, offsets, axis, _mirror)
        return 0.5 * self.norm_coef * acc


class Undecimated79:
    """Transformée bi-orthogonale 7/9 non décimée.

    Chaque échelle donne trois bandes, dans l'ordre horizontale, verticale,
    diagonale ; la dernière bande est l'échelle grossière.

    Parameters
    ----------
    n_scales : int
        Nombre d'échelles, échelle grossière comprise.
    norm : {"l1", "l2"}, default "l1"
        Normalisation des filtres. SPRITE utilise ``"l1"`` pour la
        régularisation et ``"l2"`` pour l'estimation du bruit.

    Notes
    -----
    Avec le bord miroir d'iSAP, l'inversion n'est plus exacte quand le filtre
    dilaté (``8 * 2**(n_scales - 2) + 1`` coefficients) dépasse environ deux
    fois la taille de l'image.
    """

    def __init__(self, n_scales: int, norm: str = "l1") -> None:
        self.n_scales = n_scales
        self._sb = _SubBandFilter(norm)

    @property
    def n_bands(self) -> int:
        """Nombre de bandes : ``3 * (n_scales - 1) + 1``."""
        return 3 * (self.n_scales - 1) + 1

    def transform(self, img: ArrayLike) -> NDArray[np.float64]:
        """Décompose une image.

        Parameters
        ----------
        img : array_like, shape (ny, nx)
            Image à décomposer.

        Returns
        -------
        ndarray, shape (n_bands, ny, nx)
            Bandes (horizontale, verticale, diagonale) de chaque échelle, de la
            plus fine à la plus grossière, puis l'échelle grossière.
        """
        coeffs = np.empty((self.n_bands,) + np.shape(img))
        smooth = np.asarray(img, dtype=float)
        for s in range(self.n_scales - 1):
            step = 2 ** s
            low, high = self._sb.analysis(smooth, axis=1, step=step)
            low_low, low_high = self._sb.analysis(low, axis=0, step=step)
            high_low, high_high = self._sb.analysis(high, axis=0, step=step)
            coeffs[3 * s] = high_low       # horizontale
            coeffs[3 * s + 1] = low_high   # verticale
            coeffs[3 * s + 2] = high_high  # diagonale
            smooth = low_low
        coeffs[-1] = smooth
        return coeffs

    def recons(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruit l'image (inverse iSAP).

        Parameters
        ----------
        coeffs : ndarray, shape (n_bands, ny, nx)
            Coefficients renvoyés par :meth:`transform`.

        Returns
        -------
        ndarray, shape (ny, nx)
            Image reconstruite.
        """
        img = coeffs[-1]
        for s in range(self.n_scales - 2, -1, -1):
            step = 2 ** s
            horiz, vert, diag = coeffs[3 * s:3 * s + 3]
            col_low = self._sb.synthesis(img, vert, axis=0, step=step)
            col_high = self._sb.synthesis(horiz, diag, axis=0, step=step)
            img = self._sb.synthesis(col_low, col_high, axis=1, step=step)
        return img

    def adjoint(self, coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applique l'adjoint exact de :meth:`transform`.

        Diffère de :meth:`recons` : les filtres 7/9 sont bi-orthogonaux et la
        transformée non décimée est redondante.

        Parameters
        ----------
        coeffs : ndarray, shape (n_bands, ny, nx)
            Coefficients.

        Returns
        -------
        ndarray, shape (ny, nx)
            ``Phi^T coeffs``.
        """
        g = coeffs[-1]
        for s in range(self.n_scales - 2, -1, -1):
            step = 2 ** s
            horiz, vert, diag = coeffs[3 * s:3 * s + 3]
            g_low = self._sb.analysis_adjoint(g, vert, axis=0, step=step)
            g_high = self._sb.analysis_adjoint(horiz, diag, axis=0, step=step)
            g = self._sb.analysis_adjoint(g_low, g_high, axis=1, step=step)
        return g


Transform = Starlet | Undecimated79


def make_transform(name: str | int, n_scales: int) -> Transform:
    """Construit un dictionnaire à partir de son identifiant ``run_sprite`` ou de son nom.

    Parameters
    ----------
    name : {"starlet", 2, "u79", 24}
        Dictionnaire : starlet (``-t 2``) ou 7/9 non décimée (``-t 24``).
    n_scales : int
        Nombre d'échelles.

    Returns
    -------
    Starlet or Undecimated79
        Transformée demandée.

    Raises
    ------
    ValueError
        Si le dictionnaire n'est pas reconnu.
    """
    if name in (2, "2", "starlet"):
        return Starlet(n_scales)
    if name in (24, "24", "u79", "undecimated79"):
        return Undecimated79(n_scales, norm="l1")
    raise ValueError(f"dictionnaire non reconnu {name!r} : utiliser 'starlet' (2) ou 'u79' (24)")
