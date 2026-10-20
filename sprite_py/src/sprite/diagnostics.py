"""Validation d'une reconstruction sur données réelles, sans PSF de référence.

Trois contrôles complémentaires :

1. **Ajustement** (:func:`fit_quality`) : χ²/N par image, sur les pixels
   valides. Une valeur proche de 1 indique un modèle compatible avec le bruit ;
   nettement au-dessus, le modèle n'explique pas les données (décalages, flux,
   fond, pixels défectueux non signalés) ; nettement en dessous, le niveau de
   bruit est surestimé ou la solution sur-ajuste.
2. **Prédiction d'images mises de côté** (:func:`holdout_prediction`) : les
   images sont réparties en ``k`` blocs ; chaque bloc est prédit par une
   reconstruction faite sans lui. Un χ²/N proche de celui de l'ajustement
   indique une solution qui généralise ; un écart net signale un sur-ajustement
   (régularisation trop faible) ou des paramètres d'image mal estimés.
3. **Cohérence entre moitiés** (:func:`split_half`) : deux reconstructions sur
   des images disjointes (paires et impaires) sont comparées. Leur écart mesure
   l'erreur due au bruit, pas le biais de la régularisation, commun aux deux
   moitiés : il sous-estime l'erreur vraie (d'un facteur 2 à 3 sur données
   simulées).

Pour choisir ``kappa`` : :func:`scan` applique la règle « une erreur type »
au χ²/N de prédiction (le simple minimum favorise les régularisations trop
faibles). Ces contrôles mesurent la cohérence de la solution avec les données,
pas son exactitude : un biais commun à toutes les poses leur échappe.

Les décalages et les flux sont estimés une fois sur toutes les images, puis
imposés aux reconstructions partielles, pour que les sous-ensembles restent
dans le même repère.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import fit_flux
from .metrics import compare, ellipticity, fwhm
from .operators import SpriteOperator
from .parallel import run_parallel
from .solver import SpriteConfig, SpriteResult, sprite


@dataclass
class CheckReport:
    """Résultats des contrôles de :func:`check`.

    Attributes
    ----------
    chi2 : ndarray, shape (n_frames,)
        χ²/N de l'ajustement, par image.
    holdout_chi2 : ndarray, shape (n_frames,)
        χ²/N de prédiction de chaque image par une reconstruction faite sans
        elle.
    split : dict
        Écarts entre les deux reconstructions à demi-données
        (``rel_err``, ``e1_err``, ``e2_err``, ``fwhm_err``).
    shape : dict
        FWHM (pixels HR) et ellipticités de la solution complète.
    shifts : ndarray, shape (n_frames, 2)
        Décalages ``(dy, dx)`` en pixels HR.
    flux : ndarray, shape (n_frames,)
        Flux relatifs utilisés.
    flux_dispersion : float
        Écart type relatif des flux ajustés sur la solution. Pour des flux
        égaux, il vaut 0,2 à 0,5 % sur données simulées (starlet) ; au-delà de
        3 %, les flux varient assez pour devoir être estimés.
    background : ndarray, optional
        Fond soustrait à chaque image.
    """

    chi2: NDArray[np.float64]
    holdout_chi2: NDArray[np.float64]
    split: dict[str, float]
    shape: dict[str, float]
    shifts: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_dispersion: float = 0.0
    background: NDArray[np.float64] | None = None
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        """Rapport lisible, avec les seuils d'interprétation.

        Returns
        -------
        str
            Rapport sur plusieurs lignes.
        """
        lines = [
            "Contrôle de la reconstruction SPRITE",
            "",
            f"Images : {self.chi2.size}",
            f"PSF : FWHM {self.shape['fwhm']:.3f} pixels HR, e1 {self.shape['e1']:+.4f}, e2 {self.shape['e2']:+.4f}",
            "",
            "1. Ajustement, chi2/N par image (attendu : proche de 1)",
            f"   médiane {np.median(self.chi2):.3f}, min {self.chi2.min():.3f}, max {self.chi2.max():.3f}",
            "2. Prédiction des images mises de côté, chi2/N (attendu : proche de l'ajustement)",
            f"   médiane {np.median(self.holdout_chi2):.3f}, min {self.holdout_chi2.min():.3f}, "
            f"max {self.holdout_chi2.max():.3f} ; rapport médian prédiction / ajustement "
            f"{np.median(self.holdout_chi2) / np.median(self.chi2):.3f}",
            "3. Écart entre deux reconstructions à demi-données",
            f"   relatif {self.split['rel_err']:.4f}, FWHM {100 * self.split['fwhm_err']:.2f} %, "
            f"e1 {self.split['e1_err']:.4f}, e2 {self.split['e2_err']:.4f}",
            "   (erreur due au bruit seul ; n'inclut pas le biais de la régularisation)",
            "",
            "Paramètres d'image",
            f"   décalages HR : dy de {self.shifts[:, 0].min():+.3f} à {self.shifts[:, 0].max():+.3f}, "
            f"dx de {self.shifts[:, 1].min():+.3f} à {self.shifts[:, 1].max():+.3f}",
            f"   flux relatifs : de {self.flux.min():.4f} à {self.flux.max():.4f} "
            f"(écart type {100 * self.flux.std():.2f} %)",
            f"   dispersion des flux ajustés sur la solution : {100 * self.flux_dispersion:.1f} % "
            "(0,2 à 0,5 % attendus pour des flux égaux ; au-delà de 3 %, estimer les flux)",
        ]
        if self.background is not None:
            lines.append(f"   fond soustrait : de {self.background.min():.4g} à {self.background.max():.4g}")
        if self.notes:
            lines += ["", "Points d'attention"] + [f"   - {n}" for n in self.notes]
        return "\n".join(lines)


def fit_quality(frames: NDArray[np.float64], result: SpriteResult) -> NDArray[np.float64]:
    """χ²/N de l'ajustement, par image, sur les pixels valides.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR (non utilisées directement : le résidu est dans ``result``).
    result : SpriteResult
        Reconstruction.

    Returns
    -------
    ndarray, shape (n_frames,)
        ``sum(r^2 / sigma_k^2) / N_k`` avec ``sigma_k`` le bruit estimé de
        l'image et ``N_k`` son nombre de pixels valides.
    """
    r = result.residual / result.sigma[:, None, None]
    n = result.valid.reshape(len(r), -1).sum(axis=1)
    return (np.where(result.valid, r, 0.0) ** 2).reshape(len(r), -1).sum(axis=1) / n


def _subset_config(cfg: SpriteConfig, idx: NDArray[np.int_], result: SpriteResult) -> SpriteConfig:
    up = cfg.upfact
    bad = cfg.bad_pixels
    if bad is not None and np.ndim(bad) == 3:
        bad = np.asarray(bad)[idx]
    return replace(
        cfg,
        shifts=result.shifts[idx] / up,
        fluxes=result.flux[idx] / result.flux[idx][0],
        psf_sigma=result.psf_sigma,
        ref_frame=0,
        bad_pixels=bad,
        shift_refine=0,
        flux_refine=0,
        verbose=False,
    )


def _predict(
    frames: NDArray[np.float64],
    model: NDArray[np.float64],
    hr_shifts: NDArray[np.float64],
    flux: NDArray[np.float64],
    cfg: SpriteConfig,
    background: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Images BR prédites par ``model`` (unités des données, fond ajouté)."""
    n = len(frames)
    op = SpriteOperator(hr_shifts, flux, np.ones(n), frames.shape[1:], cfg.upfact, cfg.lanczos_radius)
    pred = op.op(model)
    if background is not None:
        pred = pred + background[:, None, None]
    return pred


def _solve(frames: NDArray[np.float64], cfg: SpriteConfig) -> NDArray[np.float64]:
    """Solution d'une reconstruction partielle (tâche exécutée dans un processus)."""
    return sprite(frames, cfg).solution


def _folds(n: int, n_folds: int) -> list[NDArray[np.int_]]:
    """Blocs de test de la prédiction croisée ; chaque complément garde au moins 4 images."""
    if n < 5:
        raise ValueError("la prédiction croisée demande au moins 5 images")
    k = max(2, min(n_folds, n // 4 if n >= 8 else n))
    return [t for t in np.array_split(np.arange(n), k) if n - t.size >= 4]


def _halves(n: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Images paires et impaires."""
    if n < 8:
        raise ValueError("la comparaison entre moitiés demande au moins 8 images")
    return np.arange(0, n, 2), np.arange(1, n, 2)


def _holdout_tasks(
    frames: NDArray[np.float64], result: SpriteResult, cfg: SpriteConfig, folds: list[NDArray[np.int_]]
) -> list[tuple[NDArray[np.float64], SpriteConfig]]:
    tasks = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(frames)), test)
        tasks.append((frames[train], _subset_config(cfg, train, result)))
    return tasks


def _holdout_chi2(
    frames: NDArray[np.float64],
    result: SpriteResult,
    cfg: SpriteConfig,
    folds: list[NDArray[np.int_]],
    solutions: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    n = len(frames)
    out = np.full(n, np.nan)
    for test, model in zip(folds, solutions):
        train = np.setdiff1d(np.arange(n), test)
        bg = None if result.background is None else result.background[test]
        # Décalages imposés relatifs à l'image de référence globale : la solution
        # partielle est dans ce repère ; son échelle de flux est celle de train[0].
        flux = result.flux[test] / result.flux[train[0]]
        pred = _predict(frames[test], model, result.shifts[test], flux, cfg, bg)
        valid = result.valid[test]
        r = (frames[test] - pred) / result.sigma[test][:, None, None]
        out[test] = (np.where(valid, r, 0.0) ** 2).reshape(len(test), -1).sum(axis=1) / valid.reshape(len(test), -1).sum(axis=1)
    return out


def holdout_prediction(
    frames: NDArray[np.float64],
    result: SpriteResult,
    cfg: SpriteConfig,
    n_folds: int = 5,
    n_jobs: int | None = None,
) -> NDArray[np.float64]:
    """χ²/N de prédiction de chaque image par une reconstruction faite sans elle.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR brutes.
    result : SpriteResult
        Reconstruction complète (décalages, flux, bruit, fond, masques).
    cfg : SpriteConfig
        Paramètres de la reconstruction complète.
    n_folds : int, default 5
        Nombre de blocs ; ramené au nombre d'images si nécessaire. Chaque
        reconstruction partielle garde au moins 4 images.
    n_jobs : int, optional
        Nombre de processus (voir :func:`sprite.parallel.run_parallel`).

    Returns
    -------
    ndarray, shape (n_frames,)
        χ²/N de prédiction de chaque image, sur ses pixels valides.

    Raises
    ------
    ValueError
        S'il y a moins de 5 images.
    """
    folds = _folds(len(frames), n_folds)
    sols = run_parallel(_solve, _holdout_tasks(frames, result, cfg, folds), n_jobs)
    return _holdout_chi2(frames, result, cfg, folds, sols)


def split_half(
    frames: NDArray[np.float64], result: SpriteResult, cfg: SpriteConfig, n_jobs: int | None = None
) -> dict[str, float]:
    """Écart entre deux reconstructions sur les images paires et impaires.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images BR brutes (au moins 8 pour que chaque moitié en ait 4).
    result : SpriteResult
        Reconstruction complète.
    cfg : SpriteConfig
        Paramètres de la reconstruction complète.
    n_jobs : int, optional
        Nombre de processus.

    Returns
    -------
    dict
        Écarts de :func:`sprite.metrics.compare` entre les deux moitiés.

    Raises
    ------
    ValueError
        S'il y a moins de 8 images.
    """
    # Les deux solutions sont dans le repère de l'image de référence globale.
    xa, xb = run_parallel(_solve, [(frames[h], _subset_config(cfg, h, result)) for h in _halves(len(frames))], n_jobs)
    return compare(xb, xa)


def check(
    frames: ArrayLike, config: SpriteConfig | None = None, n_folds: int = 5, n_jobs: int | None = None,
    **kwargs: Any,
) -> tuple[SpriteResult, CheckReport]:
    """Reconstruit puis contrôle la solution par les trois tests du module.

    La reconstruction complète est faite d'abord ; les reconstructions
    partielles (blocs et moitiés) sont ensuite réparties sur ``n_jobs``
    processus.

    Parameters
    ----------
    frames : array_like, shape (n_frames, ny, nx)
        Images BR (au moins 8 pour tous les contrôles).
    config : SpriteConfig, optional
        Paramètres.
    n_folds : int, default 5
        Nombre de blocs de la prédiction croisée.
    n_jobs : int, optional
        Nombre de processus (voir :func:`sprite.parallel.run_parallel`).
    **kwargs
        Remplacent les champs de ``config``.

    Returns
    -------
    result : SpriteResult
        Reconstruction sur toutes les images.
    report : CheckReport
        Résultats des contrôles.
    """
    cfg = replace(config or SpriteConfig(), **kwargs)
    frames = np.asarray(frames, dtype=float)
    result = sprite(frames, cfg)
    folds = _folds(len(frames), n_folds)
    halves = _halves(len(frames))
    tasks = _holdout_tasks(frames, result, cfg, folds) + [(frames[h], _subset_config(cfg, h, result)) for h in halves]
    sols = run_parallel(_solve, tasks, n_jobs)
    return result, _report(frames, result, cfg, folds, sols)


def _report(
    frames: NDArray[np.float64],
    result: SpriteResult,
    cfg: SpriteConfig,
    folds: list[NDArray[np.int_]],
    sols: list[NDArray[np.float64]],
    n_groups: int = 1,
    shifts: NDArray[np.float64] | None = None,
    prefix: str = "",
) -> CheckReport:
    """Rapport à partir des solutions partielles (blocs, puis les deux moitiés).

    Avec ``n_groups > 1`` (bande d'une mosaïque de Bayer), les images sont
    rangées par sous-réseau et les indicateurs par image sont moyennés par pose.
    """
    chi2 = fit_quality(frames, result)
    hold = _holdout_chi2(frames, result, cfg, folds, sols[:len(folds)])
    split = compare(sols[-1], sols[-2])
    sol = result.solution / result.solution.sum()
    e1, e2 = ellipticity(sol)
    data = np.where(result.valid, frames, 0.0)
    if result.background is not None:
        data = data - result.background[:, None, None]
    fitted = fit_flux(data, result.solution, result.shifts, cfg.upfact, result.valid, cfg.ref_frame,
                      cfg.lanczos_radius)
    dispersion = float(np.std(fitted / np.mean(fitted)))
    flux, background = result.flux, result.background
    if n_groups > 1:
        per_pose = lambda a: None if a is None else a.reshape(n_groups, -1).mean(axis=0)  # noqa: E731
        chi2, hold, flux, background = per_pose(chi2), per_pose(hold), per_pose(flux), per_pose(background)
    notes = _notes(chi2, hold, dispersion, cfg.estimate_flux or cfg.fluxes is not None, prefix, shifts is not None)
    return CheckReport(
        chi2=chi2, holdout_chi2=hold, split=split,
        shape={"fwhm": fwhm(sol), "e1": e1, "e2": e2},
        shifts=result.shifts if shifts is None else shifts, flux=flux, flux_dispersion=dispersion,
        background=background, notes=notes,
    )


@dataclass
class BayerCheckReport:
    """Résultats de :func:`check_bayer`.

    Attributes
    ----------
    bands : dict of str to CheckReport
        Contrôles de chaque bande ; les indicateurs par image sont donnés par
        pose (moyenne des deux sous-réseaux pour G).
    pose_shifts : ndarray, shape (n_poses, 2)
        Décalages entre poses, en pixels du détecteur.
    chromatic : dict
        Décalages latéraux ``R-G`` et ``B-G`` (pixels HR) de la reconstruction
        complète.
    chromatic_split : dict
        Pour ``R-G`` et ``B-G`` : écart (norme, pixels HR) entre les décalages
        mesurés sur les deux moitiés des poses.
    general_notes : list of str
        Points d'attention communs aux bandes.
    """

    bands: dict[str, CheckReport]
    pose_shifts: NDArray[np.float64]
    chromatic: dict[str, NDArray[np.float64]]
    chromatic_split: dict[str, float]
    general_notes: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        """Points d'attention généraux, puis de chaque bande."""
        return self.general_notes + [n for r in self.bands.values() for n in r.notes]

    def text(self) -> str:
        """Rapport lisible, une section par bande.

        Returns
        -------
        str
            Rapport sur plusieurs lignes.
        """
        ps = self.pose_shifts
        lines = [
            "Contrôle de la reconstruction SPRITE, mosaïque de Bayer",
            "",
            f"Poses : {len(ps)} ; décalages (pixels du détecteur) : dy de {ps[:, 0].min():+.3f} à "
            f"{ps[:, 0].max():+.3f}, dx de {ps[:, 1].min():+.3f} à {ps[:, 1].max():+.3f}",
            "Indicateurs par pose (moyenne des deux sous-réseaux pour G) ; seuils du cas monochrome.",
            "",
            f"{'bande':<7}{'FWHM HR':>8}{'chi2 ajust.':>13}{'chi2 préd.':>12}{'rapport':>9}"
            f"{'moitiés rel':>13}{'moitiés FWHM':>14}{'disp. flux':>12}",
        ]
        for b, r in self.bands.items():
            fit, hold = np.median(r.chi2), np.nanmedian(r.holdout_chi2)
            lines.append(f"{b:<7}{r.shape['fwhm']:>8.3f}{fit:>13.3f}{hold:>12.3f}{hold / fit:>9.3f}"
                         f"{r.split['rel_err']:>13.4f}{100 * r.split['fwhm_err']:>13.2f}%"
                         f"{100 * r.flux_dispersion:>11.1f}%")
        lines.append("")
        for k in ("R-G", "B-G"):
            d = self.chromatic[k]
            lines.append(f"Décalage {k} : ({d[0]:+.3f}, {d[1]:+.3f}) pixel HR ; écart entre moitiés "
                         f"{self.chromatic_split[k]:.3f}")
        lines.append("(écarts entre moitiés : erreur due au bruit seul, biais de la régularisation exclu)")
        if self.notes:
            lines += ["", "Points d'attention"] + [f"   - {n}" for n in self.notes]
        return "\n".join(lines)


def check_bayer(
    frames: ArrayLike,
    config: SpriteConfig | None = None,
    pattern: str = "RGGB",
    pose_shifts: ArrayLike | None = None,
    pose_refine: int | None = None,
    n_folds: int = 5,
    n_jobs: int | None = None,
    **kwargs: Any,
) -> tuple[Any, BayerCheckReport]:
    """Reconstruit chaque bande d'une mosaïque de Bayer puis la contrôle.

    Les trois contrôles de :func:`check` sont appliqués à chaque bande. Les
    blocs de la prédiction croisée et les moitiés sont formés de **poses**
    entières : tous les sous-réseaux d'une pose sont mis de côté ensemble,
    comme le serait une pose réelle. Les décalages entre poses et les flux sont
    ceux de la reconstruction complète, imposés aux reconstructions partielles.
    La stabilité des décalages chromatiques ``R-G`` et ``B-G`` entre les deux
    moitiés est mesurée en plus.

    Parameters
    ----------
    frames : array_like, shape (n_poses, ny, nx)
        Mosaïques brutes (au moins 8 poses).
    config : SpriteConfig, optional
        Paramètres communs (voir :func:`sprite.bayer.sprite_bayer`).
    pattern : str, default "RGGB"
        Motif de Bayer.
    pose_shifts : array_like, shape (n_poses, 2), optional
        Décalages connus entre poses, en pixels du détecteur.
    pose_refine : int, optional
        Réajustements des décalages (voir :func:`sprite.bayer.sprite_bayer`).
    n_folds : int, default 5
        Nombre de blocs de poses de la prédiction croisée.
    n_jobs : int, optional
        Nombre de processus : les reconstructions partielles des trois bandes
        (blocs et moitiés) sont réparties entre eux.
    **kwargs
        Remplacent les champs de ``config``.

    Returns
    -------
    result : BayerResult
        Reconstruction sur toutes les poses.
    report : BayerCheckReport
        Contrôles de chaque bande.
    """
    from .bayer import _centroid, band_problems, sprite_bayer

    cfg = replace(config or SpriteConfig(), **kwargs)
    raw = np.asarray(frames, dtype=float)
    n = len(raw)
    full = sprite_bayer(raw, cfg, pattern=pattern, pose_shifts=pose_shifts, pose_refine=pose_refine, n_jobs=n_jobs)
    problems = band_problems(raw, cfg, pattern, full.pose_shifts)
    pose_folds, halves = _folds(n, n_folds), _halves(n)
    tasks, layout = [], {}
    for band, (band_frames, band_cfg) in problems.items():
        g = len(band_frames) // n
        expand = lambda idx: np.concatenate([s * n + idx for s in range(g)])  # noqa: E731
        folds = [expand(t) for t in pose_folds]
        res = full.bands[band]
        start = len(tasks)
        tasks += _holdout_tasks(band_frames, res, band_cfg, folds)
        tasks += [(band_frames[expand(h)], _subset_config(band_cfg, expand(h), res)) for h in halves]
        layout[band] = (start, len(tasks), folds, g)
    sols = run_parallel(_solve, tasks, n_jobs)

    reports, half_centroids = {}, {}
    for band, (band_frames, band_cfg) in problems.items():
        start, stop, folds, g = layout[band]
        reports[band] = _report(band_frames, full.bands[band], band_cfg, folds, sols[start:stop], n_groups=g,
                                shifts=full.pose_shifts * cfg.upfact, prefix=f"[{band}] ")
        half_centroids[band] = [_centroid(x / x.sum()) for x in sols[stop - 2:stop]]
    chromatic_split = {
        f"{b}-G": float(np.linalg.norm((half_centroids[b][0] - half_centroids["G"][0])
                                       - (half_centroids[b][1] - half_centroids["G"][1])))
        for b in ("R", "B")
    }
    general = []
    if n < 32:
        general.append(f"{n} poses : chaque moitié n'a que {n // 2} images R et B pour un sur-échantillonnage "
                       "de 4 par sous-réseau ; les indicateurs 2 et 3 de R et B sont pessimistes (sur "
                       "simulation, rapport 3 à 4 avec les décalages exacts sous 32 poses).")
    report = BayerCheckReport(bands=reports, pose_shifts=full.pose_shifts, general_notes=general,
                              chromatic={k: full.chromatic[k] for k in ("R-G", "B-G")},
                              chromatic_split=chromatic_split)
    return full, report


def _notes(
    chi2: NDArray[np.float64], hold: NDArray[np.float64], dispersion: float, flux_handled: bool,
    prefix: str = "", bayer: bool = False,
) -> list[str]:
    """Points d'attention déduits des contrôles (seuils calibrés sur simulation)."""
    notes: list[str] = []
    med = float(np.median(chi2))
    if med > 1.3:
        notes.append("chi2/N > 1,3 : le modèle n'explique pas tout le signal (fond, pixels défectueux, "
                     "saturation non signalée, flux variables, décalages).")
    if med < 0.8:
        notes.append("chi2/N < 0,8 : bruit surestimé (bruit corrélé, images prétraitées ?).")
    worst = np.argsort(chi2)[::-1][:3]
    if chi2[worst[0]] > 1.5 * med:
        notes.append(f"images atypiques (chi2/N le plus élevé) : {', '.join(str(int(i)) for i in worst)}.")
    ratio = np.nanmedian(hold) / med
    if ratio > 1.1:
        notes.append(f"prédiction {ratio:.2f} fois moins bonne que l'ajustement : régularisation trop "
                     "faible (augmenter -s) ou paramètres d'image mal estimés (décalages, flux).")
    if dispersion > 0.03 and not flux_handled:
        if bayer:
            notes.append(f"les flux ajustés varient de {100 * dispersion:.0f} % entre images : flux variables "
                         "(envisager -F --flux-method fit --flux-refine 3) ou décalages des poses imprécis "
                         "(sur simulation, 0,06 pixel d'erreur donne 3 à 7 % avec des flux égaux).")
        else:
            notes.append(f"les flux ajustés varient de {100 * dispersion:.0f} % entre images : envisager "
                         "--fluxes (flux connus) ou -F --flux-method fit --flux-refine 3.")
    return [prefix + n for n in notes]


SCANNABLE = {"kappa": float, "transform": str, "wide_factor": float, "n_reweight": int, "mask_dilate": int}


def scan(
    frames: ArrayLike,
    config: SpriteConfig,
    param: str,
    values: list[Any],
    n_folds: int = 5,
    n_jobs: int | None = None,
) -> list[dict[str, Any]]:
    """Balaye un paramètre et mesure, pour chaque valeur, l'ajustement et la prédiction croisée.

    Le χ²/N de prédiction est un critère de validation croisée, mais peu
    sensible à une régularisation trop faible : son minimum tombe souvent un
    cran sous la meilleure valeur de ``kappa``. La valeur recommandée suit donc
    la règle « une erreur type » : pour un paramètre numérique, la plus grande
    valeur dont le χ²/N de prédiction reste à moins d'une erreur type du
    minimum ; pour ``transform``, la valeur du minimum.

    Parameters
    ----------
    frames : array_like, shape (n_frames, ny, nx)
        Images BR (au moins 5).
    config : SpriteConfig
        Paramètres communs.
    param : str
        Paramètre balayé, parmi ``kappa``, ``transform``, ``wide_factor``,
        ``n_reweight``, ``mask_dilate``.
    values : list
        Valeurs essayées.
    n_folds : int, default 5
        Nombre de blocs de la prédiction croisée.
    n_jobs : int, optional
        Nombre de processus : les reconstructions complètes, puis tous les
        blocs de toutes les valeurs, sont répartis entre eux.

    Returns
    -------
    list of dict
        Pour chaque valeur : ``value``, ``chi2`` et ``holdout_chi2`` (médianes
        sur les images), ``holdout_se`` (erreur type de la médiane),
        ``ratio`` (prédiction / ajustement), ``minimum`` (valeur du minimum)
        et ``recommended`` (valeur retenue par la règle).

    Raises
    ------
    ValueError
        Si le paramètre n'est pas balayable.
    """
    if param not in SCANNABLE:
        raise ValueError(f"paramètre non balayable {param!r} ; choisir parmi {sorted(SCANNABLE)}")
    frames = np.asarray(frames, dtype=float)
    cast = SCANNABLE[param]
    cfgs = [replace(config, **{param: cast(v)}) for v in values]
    results = run_parallel(sprite, [(frames, c) for c in cfgs], n_jobs)
    folds = _folds(len(frames), n_folds)
    tasks = [t for c, r in zip(cfgs, results) for t in _holdout_tasks(frames, r, c, folds)]
    sols = run_parallel(_solve, tasks, n_jobs)
    rows = []
    for i, (value, cfg, result) in enumerate(zip(values, cfgs, results)):
        chi2 = float(np.median(fit_quality(frames, result)))
        per_frame = _holdout_chi2(frames, result, cfg, folds, sols[i * len(folds):(i + 1) * len(folds)])
        hold = float(np.median(per_frame))
        se = float(1.2533 * per_frame.std(ddof=1) / np.sqrt(per_frame.size))
        rows.append({"value": cast(value), "chi2": chi2, "holdout_chi2": hold, "holdout_se": se,
                     "ratio": hold / chi2, "minimum": False, "recommended": False})
    i_min = min(range(len(rows)), key=lambda i: rows[i]["holdout_chi2"])
    rows[i_min]["minimum"] = True
    if cast is str:
        rows[i_min]["recommended"] = True
    else:
        limit = rows[i_min]["holdout_chi2"] + rows[i_min]["holdout_se"]
        ok = [i for i in range(len(rows)) if rows[i]["holdout_chi2"] <= limit]
        rows[max(ok, key=lambda i: rows[i]["value"])]["recommended"] = True
    return rows
