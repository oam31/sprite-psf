"""Pilote de SPRITE (algorithme 2 du papier) sur le forward-backward généralisé de ModOpt.

Les valeurs par défaut sont les réglages mesurés comme les meilleurs (starlet,
kappa = 4, centroid_wide, erreurs du C++ corrigées) ; les exécutions du papier
utilisaient ``-s 4 -r 2 -F -N`` avec ``-t 2`` (starlet) ou ``-t 24`` (7/9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from modopt.opt.algorithms import GenForwardBackward
from modopt.opt.gradient import GradParent
from numpy.typing import ArrayLike, NDArray

from . import coadd, estimation
from . import shifts as shift_est
from .first_guess import first_guess
from .io import border_background
from .masking import build_masks, inpaint
from .operators import SpriteOperator
from .prox import AnalysisL1Prox, LowerBoundProx
from .wavelets import Transform, Undecimated79, default_n_scales, make_transform


@dataclass
class SpriteConfig:
    """Paramètres de SPRITE, nommés d'après les options de ``run_sprite``.

    Attributes
    ----------
    upfact : int
        Facteur de sur-échantillonnage (``-r``).
    transform : str or int
        Dictionnaire : ``"starlet"`` (2, défaut) ou ``"u79"`` (24) (``-t``) ;
        ``run_sprite`` utilisait ``"u79"`` par défaut.
    n_scales : int
        Nombre d'échelles ; ``floor(log2(N_hr)) - 1`` si négatif (``-n``).
    kappa : float
        Seuil de parcimonie en nombre d'écarts types (``-s``, défaut 4, comme
        dans le papier ; ``run_sprite`` utilisait 3 par défaut).
    n_iter : int
        Itérations maximales par passe (``-i``).
    n_reweight : int
        Passes de repondération après la première (``-R``).
    estimate_flux : bool
        Estime les flux relatifs ; sinon, flux égaux (``-F``).
    flux_method : {"aperture", "fit"}
        ``"aperture"`` : intégrale dans un disque de rayon ``4 psf_sigma``
        (C++) ; ``"fit"`` : ajustement linéaire sur la première estimation.
    flux_refine : int
        Nombre de réajustements des flux sur la solution (décalages fixes),
        chacun suivi d'une nouvelle reconstruction.
    estimate_noise_per_frame : bool
        Bruit propre à chaque image ; sinon, bruit maximal pour toutes (``-N``).
    positivity : bool
        Contrainte ``x >= 0`` (toujours active dans le C++).
    cv_win, cv_tol : int, float
        Fenêtre et tolérance du critère d'arrêt (``-C``, ``-a``).
    w1 : float
        Poids du terme de parcimonie dans le forward-backward généralisé.
    relax : float
        Relaxation ``lambda`` du forward-backward généralisé.
    lanczos_radius : int
        Rayon des noyaux de recalage de l'opérateur.
    first_guess_nsig : float
        Seuil du débruitage de la première estimation.
    shift_nsig : float
        Seuil de bruit pour l'estimation des décalages.
    shift_method : str, optional
        ``"centroid_wide"`` (défaut), ``"centroid"`` (C++) ou ``"xcorr"`` ;
        ``None`` choisit ``centroid_wide``, ou ``centroid`` si ``cpp_compat``.
    wide_factor : float
        Fenêtre de ``centroid_wide``, en unités de ``psf_sigma``.
    shift_refine : int
        Nombre de réajustements des décalages (et des flux) sur la solution.
    subtract_background : bool
        Soustrait la médiane de la bordure de chaque image.
    saturation : float, optional
        Niveau de saturation en unités des données brutes ; les pixels qui
        l'atteignent deviennent des bornes inférieures (terme censuré).
    bad_pixels : array_like of bool, optional
        Pixels défectueux connus (``True`` = défectueux), ``(ny, nx)`` ou
        ``(n_frames, ny, nx)``.
    mask_dilate : int
        Extension, en pixels, des zones invalides.
    ref_frame : int
        Image de référence (décalage nul, flux 1).
    shifts : array_like, optional
        Décalages imposés ``(dy, dx)`` en pixels BR (``-S``).
    fluxes : array_like, optional
        Flux relatifs imposés (temps de pose connus, par exemple) ; prioritaires
        sur ``estimate_flux``.
    psf_sigma : float, optional
        Largeur de PSF imposée, en pixels BR.
    spectral_radius : float, optional
        Rayon spectral imposé.
    exact_adjoint : bool, optional
        Adjoint exact du dictionnaire dans le prox d'analyse (``True``) ou
        reconstruction de la transformée comme le C++ (``False``). ``None``
        choisit selon les mesures : adjoint pour la starlet (le C++ y laisse
        un fond étendu), reconstruction pour la 7/9 (l'adjoint exact y donne
        des résultats légèrement moins bons) ; reconstruction si ``cpp_compat``.
    cpp_compat : bool
        **Rétrocompatibilité** : reproduit les erreurs du C++ publié (recalage
        de la première estimation, ouverture de flux, critère d'arrêt,
        reconstruction à la place de l'adjoint dans le prox) et son estimateur
        de décalages ``centroid``, pour retrouver ses résultats à l'identique.
        À ne pas utiliser pour des mesures.
    compat_stack, compat_flux, compat_stopping : bool, optional
        **Rétrocompatibilité** : reproduit une seule de ces erreurs ; ``None``
        suit ``cpp_compat``.
    verbose : bool
        Affiche les étapes.
    """

    # --- Paramètres principaux ---
    transform: str | int = "starlet"
    kappa: float = 4.0
    estimate_noise_per_frame: bool = False
    upfact: int = 2
    # --- Décalages ---
    shift_method: str | None = None
    wide_factor: float = 2.5
    shifts: ArrayLike | None = None
    shift_refine: int = 0
    # --- Flux ---
    estimate_flux: bool = False
    flux_method: str = "aperture"
    flux_refine: int = 0
    fluxes: ArrayLike | None = None
    # --- Fond et pixels ---
    subtract_background: bool = False
    saturation: float | None = None
    bad_pixels: ArrayLike | None = None
    mask_dilate: int = 0
    # --- Algorithme (avancé) ---
    n_scales: int = -1
    n_iter: int = 50
    n_reweight: int = 1
    cv_win: int = 25
    cv_tol: float = 0.005
    exact_adjoint: bool | None = None
    positivity: bool = True
    w1: float = 0.5
    relax: float = 1.4
    lanczos_radius: int = 10
    first_guess_nsig: float = 5.0
    shift_nsig: float = 4.0
    ref_frame: int = 0
    psf_sigma: float | None = None
    spectral_radius: float | None = None
    # --- Rétrocompatibilité avec le C++ publié : reproduit des erreurs ---
    cpp_compat: bool = False
    compat_stack: bool | None = None
    compat_flux: bool | None = None
    compat_stopping: bool | None = None
    verbose: bool = False

    def resolved_shift_method(self) -> str:
        """Estimateur effectif : ``centroid_wide`` par défaut, ``centroid`` en mode compatible."""
        if self.shift_method is not None:
            return self.shift_method
        return "centroid" if self.cpp_compat else "centroid_wide"

    def resolved_exact_adjoint(self) -> bool:
        """Adjoint exact pour la starlet, reconstruction pour la 7/9 et en mode compatible."""
        if self.exact_adjoint is not None:
            return self.exact_adjoint
        if self.cpp_compat:
            return False
        return self.transform in (2, "2", "starlet")

    def compat(self, which: str) -> bool:
        """Indique si le défaut ``which`` (``stack``, ``flux``, ``stopping``) du C++ est reproduit."""
        value = getattr(self, f"compat_{which}")
        return self.cpp_compat if value is None else value


@dataclass
class SpriteResult:
    """Résultat d'une reconstruction.

    Attributes
    ----------
    solution : ndarray, shape (ny * upfact, nx * upfact)
        PSF HR, alignée sur l'image de référence.
    first_guess : ndarray
        Première estimation ``x0``.
    centroids : ndarray, shape (n_frames, 2)
        Centroïdes ``(y, x)`` comptés à partir de 1.
    shifts : ndarray, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels HR.
    flux : ndarray, shape (n_frames,)
        Flux relatifs.
    sigma : ndarray, shape (n_frames,)
        Niveau de bruit estimé de chaque image.
    spectral_radius : float
        Rayon spectral de ``M^T M``.
    psf_sigma : float
        Largeur de PSF utilisée, en pixels BR.
    residual : ndarray, shape (n_frames, ny, nx)
        Résidu BR en unités des données, nul sur les pixels invalides.
    lr_model : ndarray, shape (n_frames, ny, nx)
        Images BR prédites par la solution, en unités des données.
    valid : ndarray of bool, shape (n_frames, ny, nx)
        Pixels pris en compte normalement dans l'attache aux données.
    censored : ndarray of bool, shape (n_frames, ny, nx)
        Pixels saturés, pris en compte comme bornes inférieures.
    residual_energy : list of float
        Énergie du résidu blanchi à chaque itération.
    n_iter : int
        Itérations de la dernière passe.
    background : ndarray, optional
        Fond soustrait à chaque image.
    shift_history : list of ndarray
        Décalages HR de chaque tour de réajustement.
    """

    solution: NDArray[np.float64]
    first_guess: NDArray[np.float64]
    centroids: NDArray[np.float64]
    shifts: NDArray[np.float64]
    flux: NDArray[np.float64]
    sigma: NDArray[np.float64]
    spectral_radius: float
    psf_sigma: float
    residual: NDArray[np.float64]
    lr_model: NDArray[np.float64]
    valid: NDArray[np.bool_]
    censored: NDArray[np.bool_]
    residual_energy: list[float] = field(default_factory=list)
    n_iter: int = 0
    background: NDArray[np.float64] | None = None
    shift_history: list[NDArray[np.float64]] = field(default_factory=list)


class SpriteGradient(GradParent):
    """Gradient de l'attache aux données avec pixels exclus et censurés.

    Pour ``r = y_w - M(x0 + delta)`` :

    ``J1 = 1/2 sum_valides r^2 + 1/2 sum_saturés max(r, 0)^2``

    Un pixel exclu n'intervient pas ; un pixel saturé n'est pénalisé que si le
    modèle passe sous la valeur mesurée. ``J1`` reste convexe, de gradient
    lipschitzien de même constante que sans masque.

    Parameters
    ----------
    data_w : ndarray, shape (n_frames, ny, nx)
        Images blanchies.
    operator : SpriteOperator
        Opérateur ``M``.
    x0 : ndarray, shape (ny * upfact, nx * upfact)
        Première estimation.
    valid : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels valides ; tous par défaut.
    censored : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels saturés ; aucun par défaut.
    """

    def __init__(
        self,
        data_w: NDArray[np.float64],
        operator: SpriteOperator,
        x0: NDArray[np.float64],
        valid: NDArray[np.bool_] | None = None,
        censored: NDArray[np.bool_] | None = None,
    ) -> None:
        self.x0 = x0
        self.valid = np.ones(data_w.shape, bool) if valid is None else valid
        self.censored = np.zeros(data_w.shape, bool) if censored is None else censored
        self.residual: NDArray[np.float64] | None = None
        used = self.valid | self.censored
        super().__init__(
            np.where(used, data_w, 0.0), operator.op, operator.adj_op,
            cost=lambda d: 0.5 * float(np.sum(self.residual ** 2)), verbose=False,
        )
        self.get_grad = self._get_grad

    def _get_grad(self, delta: NDArray[np.float64]) -> None:
        r = self.obs_data - self.op(delta + self.x0)
        r = np.where(self.censored, np.maximum(r, 0.0), r)
        self.residual = np.where(self.valid | self.censored, r, 0.0)
        self._grad = -self.trans_op(self.residual)


class _StoppedGFB(GenForwardBackward):
    """Forward-backward généralisé avec le critère d'arrêt à fenêtre glissante du C++.

    L'arrêt intervient quand la moyenne de ``sum |delta_{n+1} - delta_n|`` sur
    ``cv_win`` itérations passe sous ``cv_tol * sum |delta_n|``.
    """

    def __init__(
        self,
        *args: Any,
        cv_win: int,
        cv_tol: float,
        diff_buffer: NDArray[np.float64],
        cpp_compat: bool,
        on_iter: Callable[[], None],
        **kwargs: Any,
    ) -> None:
        self._cv_win = cv_win
        self._cv_tol = cv_tol
        self._buf = diff_buffer
        self._cpp_compat = cpp_compat
        self._on_iter = on_iter
        self._mean_diff = 0.0
        super().__init__(*args, **kwargs)

    def _update(self) -> None:
        prev = self._x_old.copy()
        super()._update()
        self._on_iter()
        m = self.idx
        diff = np.abs(self._x_new - prev).sum()
        old_l1 = np.abs(prev).sum()
        win = self._cv_win
        # Dans le C++, le tampon n'est pas remis à zéro entre les passes.
        leaving = self._buf[m % win] if m >= win else 0.0
        self._buf[m % win] = diff
        self._mean_diff += diff / win
        if m >= win:
            if self._cpp_compat:
                # C++ : retire buffer[(m + 1) % win], encore dans la fenêtre
                self._mean_diff -= self._buf[(m + 1) % win] / win
            else:
                self._mean_diff -= leaving / win
            self.converge = self._mean_diff < self._cv_tol * old_l1


def _log(cfg: SpriteConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def sprite(frames: ArrayLike, config: SpriteConfig | None = None, **kwargs: Any) -> SpriteResult:
    """Reconstruit une PSF HR à partir d'images BR décalées.

    Parameters
    ----------
    frames : array_like, shape (n_frames, ny, nx)
        Images BR d'une même source ponctuelle.
    config : SpriteConfig, optional
        Paramètres ; valeurs par défaut sinon.
    **kwargs
        Remplacent les champs correspondants de ``config``.

    Returns
    -------
    SpriteResult
        Solution, paramètres estimés et diagnostics.

    Raises
    ------
    TypeError
        Si un paramètre de ``kwargs`` n'existe pas.

    Examples
    --------
    >>> res = sprite(frames, kappa=4, estimate_noise_per_frame=True)  # doctest: +SKIP
    >>> res.solution.shape  # doctest: +SKIP
    (168, 168)
    """
    cfg = config or SpriteConfig()
    for key, val in kwargs.items():
        if not hasattr(cfg, key):
            raise TypeError(f"paramètre SPRITE inconnu {key!r}")
        setattr(cfg, key, val)

    raw = np.asarray(frames, dtype=float)
    nz, ny, nx = raw.shape
    up = cfg.upfact
    n_hr = min(ny, nx) * up
    n_scales = cfg.n_scales if cfg.n_scales > 0 else default_n_scales(n_hr)
    dictionary = make_transform(cfg.transform, n_scales)
    mod_transform = Undecimated79(default_n_scales(n_hr))

    # --- Pixels valides, fond ---
    valid, censored = build_masks(raw, cfg.saturation, cfg.bad_pixels, cfg.mask_dilate)
    usable = valid | censored
    data = np.where(usable, raw, 0.0)
    background = None
    if cfg.subtract_background:
        background = border_background(raw, valid=valid)
        data = np.where(usable, data - background[:, None, None], 0.0)
    # Images complètes pour les estimations préalables : pixels exclus interpolés,
    # pixels saturés gardés à leur valeur écrêtée.
    est = data if usable.all() else inpaint(data, usable)

    # --- Paramètres d'attache aux données (section 3.4.1) ---
    sigma = estimation.noise_sigma(est)
    weights = sigma if cfg.estimate_noise_per_frame else np.full(nz, sigma.max())

    psf_sig = cfg.psf_sigma
    if psf_sig is None:
        _log(cfg, "Estimation de la taille de la PSF...")
        psf_sig = coadd.psf_sigma(est, up, mod_transform, cfg.first_guess_nsig)

    method = cfg.resolved_shift_method()
    _log(cfg, f"Estimation des décalages ({method})...")
    lr_shifts, cents = shift_est.estimate_shifts(
        est, sigma, psf_sig, method, cfg.ref_frame, cfg.shift_nsig, cfg.wide_factor)
    if cfg.shifts is not None:
        lr_shifts = np.asarray(cfg.shifts, dtype=float)

    _log(cfg, "Estimation des flux...")
    flux = np.ones(nz)
    if cfg.estimate_flux:
        if cfg.flux_method == "aperture":
            flux = estimation.estimate_flux(
                est, cents, 4 * psf_sig, cfg.ref_frame, cpp_compat=cfg.compat("flux"))
        elif cfg.flux_method == "fit":
            # Modèle à flux égaux, puis ajustement ; deux passes pour que le
            # modèle soit construit avec des images déjà renormalisées.
            for _ in range(2):
                x0 = first_guess(est / flux[:, None, None], up, lr_shifts, mod_transform,
                                 cfg.first_guess_nsig, cfg.compat("stack"))
                flux = estimation.fit_flux(est, x0, up * lr_shifts, up, valid, cfg.ref_frame,
                                           cfg.lanczos_radius)
        else:
            raise ValueError(f"méthode de flux inconnue {cfg.flux_method!r} : 'aperture' ou 'fit'")
    if cfg.fluxes is not None:
        flux = np.asarray(cfg.fluxes, dtype=float)
        flux = flux / flux[cfg.ref_frame]

    history: list[NDArray[np.float64]] = []
    out: dict[str, Any] = {}
    for rnd in range(cfg.shift_refine + 1):
        if rnd > 0:
            _log(cfg, f"Réajustement {rnd} des décalages sur la solution...")
            lr_shifts, fitted_flux = shift_est.refine_on_model(
                data, sigma, out["solution"], up, lr_shifts, flux, cfg.ref_frame, valid)
            if cfg.estimate_flux:
                flux = fitted_flux
        history.append(up * lr_shifts)
        out = _reconstruct(data, est, valid, censored, cfg, lr_shifts, flux, weights, dictionary, mod_transform)
    for rnd in range(cfg.flux_refine if cfg.estimate_flux and cfg.fluxes is None else 0):
        _log(cfg, f"Réajustement {rnd + 1} des flux sur la solution...")
        flux = estimation.fit_flux(data, out["solution"], up * lr_shifts, up, valid, cfg.ref_frame,
                                   cfg.lanczos_radius)
        out = _reconstruct(data, est, valid, censored, cfg, lr_shifts, flux, weights, dictionary, mod_transform)

    return SpriteResult(
        solution=out["solution"], first_guess=out["x0"], centroids=cents, shifts=up * lr_shifts,
        flux=flux, sigma=sigma, spectral_radius=out["rho"], psf_sigma=psf_sig,
        residual=out["residual"], lr_model=out["lr_model"], valid=valid, censored=censored,
        residual_energy=out["energy"], n_iter=out["n_iter"],
        background=background, shift_history=history,
    )


def _reconstruct(
    data: NDArray[np.float64],
    est: NDArray[np.float64],
    valid: NDArray[np.bool_],
    censored: NDArray[np.bool_],
    cfg: SpriteConfig,
    lr_shifts: NDArray[np.float64],
    flux: NDArray[np.float64],
    weights: NDArray[np.float64],
    dictionary: Transform,
    mod_transform: Transform,
) -> dict[str, Any]:
    """Première estimation puis forward-backward généralisé repondéré, décalages et flux donnés."""
    nz, ny, nx = data.shape
    up = cfg.upfact
    hr_shape = (ny * up, nx * up)

    _log(cfg, "Calcul de la première estimation...")
    # Images ramenées au flux de la référence avant l'empilement (le C++ empile
    # les images brutes, même quand leurs flux diffèrent).
    stack_in = est if cfg.compat("flux") else est / flux[:, None, None]
    x0 = first_guess(stack_in, up, lr_shifts, mod_transform, cfg.first_guess_nsig, cfg.compat("stack"))
    if cfg.positivity:
        x0 = np.maximum(x0, 0.0)

    # --- Opérateur et pas (section 3.4.4) ---
    operator = SpriteOperator(up * lr_shifts, flux, weights, (ny, nx), up, cfg.lanczos_radius)
    rho = cfg.spectral_radius
    if rho is None:
        # Sans masque : majorant du rayon spectral masqué, donc pas sûr.
        _log(cfg, "Estimation du rayon spectral...")
        rho = operator.spectral_radius()[0]
    step = 1.0 / rho

    data_w = data / weights[:, None, None]
    grad = SpriteGradient(data_w, operator, x0, valid, censored)
    sparsity = AnalysisL1Prox(dictionary, grad, step, cfg.kappa, hr_shape,
                              exact_adjoint=cfg.resolved_exact_adjoint())
    prox_list = [sparsity, LowerBoundProx(x0)]

    energy: list[float] = []
    delta = np.zeros(hr_shape)
    z = None
    diff_buffer = np.zeros(cfg.cv_win)
    n_done = 0
    _log(cfg, "Minimisation...")
    for rw in range(cfg.n_reweight + 1):
        _log(cfg, f"Passe {rw}")
        sparsity.start_pass()
        if not cfg.compat("stopping"):
            diff_buffer[:] = 0.0
        alg = _StoppedGFB(
            delta, grad, prox_list, cost=None,
            gamma_param=step, lambda_param=cfg.relax, weights=[cfg.w1, 1 - cfg.w1],
            auto_iterate=False, progress=False, verbose=False,
            cv_win=cfg.cv_win, cv_tol=cfg.cv_tol, diff_buffer=diff_buffer,
            cpp_compat=cfg.compat("stopping"),
            on_iter=lambda: energy.append(float(np.sum(grad.residual ** 2))),
        )
        if z is not None:
            alg._z = z
        alg.iterate(max_iter=cfg.n_iter)
        delta, z = alg.x_final.copy(), alg._z
        n_done = alg.idx + 1
        sparsity.reweight(delta)

    solution = delta + x0
    return {
        "solution": solution, "x0": x0, "rho": rho, "energy": energy, "n_iter": n_done,
        "residual": grad.residual * weights[:, None, None],
        "lr_model": operator.op(solution) * weights[:, None, None],
    }
