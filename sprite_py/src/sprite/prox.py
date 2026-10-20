"""Opérateurs proximaux de SPRITE, sous forme de ``ProximityParent`` de ModOpt.

* :class:`AnalysisL1Prox` : terme ℓ1 d'analyse pondéré
  ``kappa || w . lambda . Phi delta ||_1`` (éq. 20 du papier). Avec un
  dictionnaire redondant, il n'a pas de forme close ; il est calculé par FISTA
  sur le problème dual (éq. 21, C++ ``wl_filter_analysis``). Les niveaux de
  bruit par coefficient ``lambda`` sont réestimés à chaque itération externe à
  partir du pas de gradient courant (C++ ``noise_est_meth = 0``).
* :class:`LowerBoundProx` : projection sur ``delta >= -x0`` (positivité de
  ``x = x0 + delta``, éq. 18).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from modopt.opt.proximity import ProximityParent
from numpy.typing import NDArray

from .wavelets import Transform, mad_sigma

if TYPE_CHECKING:
    from .solver import SpriteGradient


def soft_threshold(x: NDArray[np.float64], thresh: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Seuillage doux au sens d'iSAP (``thresholding`` avec ``thresh_type`` doux).

    Parameters
    ----------
    x : ndarray
        Valeurs.
    thresh : float or ndarray
        Seuil, scalaire ou de même forme que ``x``.

    Returns
    -------
    ndarray
        ``0`` si ``|x| < thresh``, sinon ``x - thresh sign(x)``.
    """
    return np.where(np.abs(x) < thresh, 0.0, x - thresh * np.sign(x))


def analysis_prox(
    img: NDArray[np.float64],
    transform: Transform,
    thresh: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    n_iter: int,
    mu: float = 0.5,
    exact_adjoint: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calcule ``argmin_x 1/2 ||img - x||^2 + || thresh . Phi x ||_1`` (``wl_filter_analysis``).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Point où évaluer le prox.
    transform : Starlet or Undecimated79
        Dictionnaire ``Phi``.
    thresh : ndarray, shape (n_bands - 1, ny, nx)
        Seuils de chaque coefficient de détail ; l'échelle grossière n'est pas
        pénalisée (sa variable duale est forcée à zéro).
    coeffs : ndarray, shape (n_bands, ny, nx)
        Variable duale de départ (démarrage à chaud).
    n_iter : int
        Nombre d'itérations FISTA.
    mu : float, default 0.5
        Pas des itérations duales ; convergent si ``mu < 2 / ||Phi||^2``
        (``||Phi||^2`` vaut 1,33 pour la starlet et 1,81 pour la 7/9).
    exact_adjoint : bool, default True
        Utilise l'adjoint exact ``Phi^T``, comme l'exige l'algorithme dual.
        ``False`` reproduit le C++, qui utilise la reconstruction de la
        transformée : l'opérateur calculé n'est alors plus le prox de la
        norme d'analyse (fond positif étendu avec la starlet).

    Returns
    -------
    x : ndarray, shape (ny, nx)
        Valeur du prox.
    coeffs : ndarray, shape (n_bands, ny, nx)
        Variable duale finale, à réutiliser au prochain appel.
    """
    back = transform.adjoint if exact_adjoint else transform.recons
    z = coeffs.copy()
    x_old = np.zeros_like(coeffs)
    t = 1.0
    for _ in range(n_iter):
        resid = img - back(z)
        x = z + mu * transform.transform(resid)
        # x - soft(x) : projection sur la boule l-infini pondérée
        x[:-1] -= soft_threshold(x[:-1], thresh)
        x[-1] = 0.0
        t_new = (1 + np.sqrt(4 * t ** 2 + 1)) / 2
        lam = 1 + (t - 1) / t_new
        t = t_new
        z = x_old + lam * (x - x_old)
        x_old = x
    return img - back(z), z


class AnalysisL1Prox(ProximityParent):
    """Opérateur proximal de la parcimonie d'analyse repondérée, pour le forward-backward généralisé.

    Parameters
    ----------
    transform : Starlet or Undecimated79
        Dictionnaire ``Phi``.
    grad : SpriteGradient
        Opérateur de gradient ; sa valeur courante fixe les niveaux de bruit.
    step : float
        Pas de gradient ``gamma = 1 / rho(M^T M)``.
    kappa : float
        Paramètre de parcimonie (option ``-s`` ; 4 dans le papier).
    shape : tuple of int
        Taille de l'image HR.
    exact_adjoint : bool, default True
        Adjoint exact dans le calcul dual ; ``False`` reproduit le C++.
    """

    def __init__(
        self,
        transform: Transform,
        grad: SpriteGradient,
        step: float,
        kappa: float,
        shape: tuple[int, int],
        exact_adjoint: bool = True,
    ) -> None:
        self.exact_adjoint = exact_adjoint
        self.transform = transform
        self.grad = grad
        self.step = step
        self.kappa = kappa
        n_det = transform.n_bands - 1
        self.weights = np.ones((n_det,) + shape)
        self.noise_map = np.zeros((n_det,) + shape)
        self.coeffs = np.zeros((transform.n_bands,) + shape)
        self.n_subiter = 100
        super().__init__(self._prox, self._cost)

    def start_pass(self) -> None:
        """Remet à 100 le nombre de sous-itérations, comme le C++ à chaque passe de repondération."""
        self.n_subiter = 100

    def update_noise(self) -> None:
        """Réestime le bruit de chaque bande : MAD de ``Phi(-gamma grad)`` (``wl_gnoise_est``)."""
        coeffs = self.transform.transform(-self.step * self.grad.grad)
        for b in range(self.noise_map.shape[0]):
            self.noise_map[b] = mad_sigma(coeffs[b])

    def reweight(self, delta: NDArray[np.float64]) -> None:
        """Met à jour les poids ``w = 1 / (1 + |Phi delta| / (3 sigma))`` (algorithme 2, étape 8).

        Parameters
        ----------
        delta : ndarray, shape (ny, nx)
            Correction courante.
        """
        coeffs = self.transform.transform(delta)[:-1]
        self.weights = 1.0 / (1.0 + np.abs(coeffs) / (3.0 * self.noise_map))

    def _prox(self, x: NDArray[np.float64], extra_factor: float = 1.0) -> NDArray[np.float64]:
        self.update_noise()
        # extra_factor = gamma / w_i : la carte de bruit contient déjà gamma.
        thresh = self.kappa * (self.weights * self.noise_map) * (extra_factor / self.step)
        out, self.coeffs = analysis_prox(x, self.transform, thresh, self.coeffs, self.n_subiter,
                                         exact_adjoint=self.exact_adjoint)
        self.n_subiter = 5
        return out

    def _cost(self, x: NDArray[np.float64]) -> float:
        coeffs = self.transform.transform(x)[:-1]
        return float(self.kappa * np.abs(self.weights * self.noise_map * coeffs).sum())


class LowerBoundProx(ProximityParent):
    """Projection sur ``{delta : delta >= -x0}``.

    Parameters
    ----------
    x0 : ndarray, shape (ny, nx)
        Première estimation.
    """

    def __init__(self, x0: NDArray[np.float64]) -> None:
        self.lower = -x0
        super().__init__(self._project, lambda x: 0.0)

    def _project(self, x: NDArray[np.float64], extra_factor: float = 1.0) -> NDArray[np.float64]:
        return np.maximum(x, self.lower)
