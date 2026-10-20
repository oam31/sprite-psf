"""Reconstruction de PSF par bande à partir d'images brutes en mosaïque de Bayer.

Une image Bayer brute (non dématricée) échantillonne chaque bande sur un
sous-réseau du détecteur. Avec le motif RGGB, sur chaque bloc de 2 × 2 pixels :

* R en (0, 0), G en (0, 1) et (1, 0), B en (1, 1).

Chaque sous-réseau a un pas de 2 pixels : R et B sont deux fois plus
sous-échantillonnés qu'avec une matrice monochrome ; G l'est un peu moins
grâce à ses deux sous-réseaux. Le module traite chaque bande comme un problème
SPRITE indépendant (PSF propre à chaque bande, effets chromatiques compris) :

1. chaque pose est découpée en sous-images (R, G1, G2, B), une par
   sous-réseau, de pas 2 pixels et de décalage connu ;
2. les décalages entre poses, achromatiques, sont estimés une seule fois sur
   les sous-images vertes, puis appliqués à toutes les bandes ;
3. chaque bande est reconstruite par :func:`sprite.sprite` sur une grille HR
   commune, de pas ``1 / upfact`` pixel du détecteur. Les deux sous-réseaux
   verts entrent comme deux images par pose.

Les décalages étant communs, un décalage chromatique latéral apparaît comme
une translation entre les PSF reconstruites : il est mesuré, pas imposé.

Il ne faut pas dématricer les images avant : le dématriçage interpole les
pixels manquants, ce qui fausse le modèle d'observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import estimation
from .coadd import psf_sigma as estimate_psf_sigma
from .metrics import ellipticity, fwhm
from .parallel import run_parallel
from .shifts import estimate_shifts
from .solver import SpriteConfig, SpriteResult, sprite
from .wavelets import Undecimated79, default_n_scales

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


def sublattices(pattern: str = "RGGB") -> dict[str, list[tuple[int, int]]]:
    """Positions ``(ligne, colonne)`` de chaque bande dans le bloc 2 × 2.

    Parameters
    ----------
    pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}, default "RGGB"
        Motif lu ligne par ligne à partir du pixel (0, 0).

    Returns
    -------
    dict
        ``{"R": [(y, x)], "G": [(y1, x1), (y2, x2)], "B": [(y, x)]}``.

    Raises
    ------
    ValueError
        Si le motif n'est pas reconnu.
    """
    pattern = pattern.upper()
    if pattern not in PATTERNS:
        raise ValueError(f"motif de Bayer inconnu {pattern!r} ; choisir parmi {PATTERNS}")
    pos = {"R": [], "G": [], "B": []}
    for i, c in enumerate(pattern):
        pos[c].append((i // 2, i % 2))
    return pos


def split_bayer(frames: ArrayLike, pattern: str = "RGGB") -> dict[str, list[tuple[NDArray[np.float64], tuple[int, int]]]]:
    """Découpe des poses brutes en sous-images par sous-réseau.

    Parameters
    ----------
    frames : array_like, shape (n_poses, ny, nx)
        Images brutes en mosaïque ; ``ny`` et ``nx`` doivent être pairs.
    pattern : str, default "RGGB"
        Motif de Bayer.

    Returns
    -------
    dict
        Pour chaque bande, la liste des ``(sous-images, (oy, ox))`` : un
        tableau ``(n_poses, ny / 2, nx / 2)`` par sous-réseau et sa position
        dans le bloc 2 × 2.

    Raises
    ------
    ValueError
        Si la taille des images est impaire.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.shape[1] % 2 or frames.shape[2] % 2:
        raise ValueError(f"images de taille impaire {frames.shape[1:]} : la mosaïque doit être complète")
    return {band: [(frames[:, oy::2, ox::2], (oy, ox)) for oy, ox in offs]
            for band, offs in sublattices(pattern).items()}


def lattice_shifts(pose_shifts: NDArray[np.float64], offset: tuple[int, int]) -> NDArray[np.float64]:
    """Décalages, en pixels du sous-réseau, des sous-images d'un sous-réseau.

    Le sous-réseau de position ``o`` échantillonne le pixel ``2 i + o`` ; une
    pose décalée de ``s`` pixels du détecteur y apparaît décalée de
    ``(s - o) / 2`` pixels du sous-réseau, par rapport au sous-réseau (0, 0) de
    la pose de référence.

    Parameters
    ----------
    pose_shifts : ndarray, shape (n_poses, 2)
        Décalages ``(dy, dx)`` des poses, en pixels du détecteur.
    offset : tuple of int
        Position ``(oy, ox)`` du sous-réseau dans le bloc 2 × 2.

    Returns
    -------
    ndarray, shape (n_poses, 2)
        Décalages en pixels du sous-réseau.
    """
    return (np.asarray(pose_shifts, dtype=float) - np.asarray(offset, dtype=float)) / 2.0


def _luminance(frames: NDArray[np.float64], pattern: str) -> NDArray[np.float64]:
    """Mosaïque dont chaque bande est divisée par son flux moyen (pseudo-image monochrome)."""
    lum = np.array(frames, dtype=float)
    for offs in sublattices(pattern).values():
        scale = sum(lum[:, oy::2, ox::2].sum() for oy, ox in offs) / len(offs)
        for oy, ox in offs:
            lum[:, oy::2, ox::2] /= scale
    return lum


def estimate_pose_shifts(
    frames: ArrayLike, pattern: str = "RGGB", cfg: SpriteConfig | None = None
) -> NDArray[np.float64]:
    """Décalages achromatiques entre poses, première estimation sur la pseudo-luminance.

    Chaque bande de la mosaïque est divisée par son flux moyen, puis
    l'estimateur de décalages de ``cfg`` (``centroid_wide`` par défaut)
    s'applique à cette pseudo-image monochrome. La précision reste limitée
    (environ 0,1 pixel sur données simulées) : le mélange des couleurs et les
    différences de PSF entre bandes biaisent le centroïde. :func:`refine_pose_shifts`
    l'affine ensuite sur le modèle reconstruit.

    Parameters
    ----------
    frames : array_like, shape (n_poses, ny, nx)
        Images brutes en mosaïque (fond déjà soustrait si nécessaire).
    pattern : str, default "RGGB"
        Motif de Bayer.
    cfg : SpriteConfig, optional
        Paramètres (estimateur, fenêtre, seuil).

    Returns
    -------
    ndarray, shape (n_poses, 2)
        Décalages ``(dy, dx)`` en pixels du détecteur, nuls pour la pose 0.
    """
    cfg = cfg or SpriteConfig()
    lum = _luminance(np.asarray(frames, dtype=float), pattern)
    sigma = estimation.noise_sigma(lum)
    n_hr = min(lum.shape[1:]) * cfg.upfact
    psf_sig = estimate_psf_sigma(lum, cfg.upfact, Undecimated79(default_n_scales(n_hr)), cfg.first_guess_nsig)
    shifts, _ = estimate_shifts(lum, sigma, psf_sig, cfg.resolved_shift_method(), 0, cfg.shift_nsig, cfg.wide_factor)
    return shifts


def refine_pose_shifts(
    frames: NDArray[np.float64],
    models: dict[str, NDArray[np.float64]],
    pose_shifts: NDArray[np.float64],
    upfact: int,
    pattern: str = "RGGB",
    valid: NDArray[np.bool_] | None = None,
    half: int = 12,
) -> NDArray[np.float64]:
    """Réajuste le décalage de chaque pose sur toute la mosaïque, avec la PSF de chaque bande.

    Pour chaque pose, le décalage ``s`` (commun aux bandes) et un flux par
    bande minimisent l'écart entre la mosaïque et sa prédiction : chaque
    pixel est prédit par la PSF HR de sa bande, décalée de ``upfact s`` pixels
    HR puis échantillonnée au pas du détecteur.

    Parameters
    ----------
    frames : ndarray, shape (n_poses, ny, nx)
        Mosaïques (fond soustrait).
    models : dict of str to ndarray
        PSF HR de chaque bande, de taille ``(upfact ny, upfact nx)``, alignées
        sur la pose de référence.
    pose_shifts : ndarray, shape (n_poses, 2)
        Décalages initiaux, en pixels du détecteur.
    upfact : int
        Facteur de sur-échantillonnage par rapport au détecteur.
    pattern : str, default "RGGB"
        Motif de Bayer.
    valid : ndarray of bool, shape (n_poses, ny, nx), optional
        Pixels utilisables.
    half : int, default 12
        Demi-taille, en pixels du détecteur, de la fenêtre d'ajustement
        centrée sur l'étoile.

    Returns
    -------
    ndarray, shape (n_poses, 2)
        Décalages réajustés, relatifs à la pose 0, en pixels du détecteur.
    """
    from scipy.optimize import least_squares

    from .shifts import fourier_shift

    n, ny, nx = frames.shape
    pos = sublattices(pattern)
    bands = list(pos)
    band_map = np.empty((ny, nx), dtype=int)
    for i, b in enumerate(bands):
        for oy, ox in pos[b]:
            band_map[oy::2, ox::2] = i
    cy, cx = np.unravel_index(np.argmax(sum(models.values())), models[bands[0]].shape)
    cy, cx = cy // upfact, cx // upfact
    win = np.zeros((ny, nx), bool)
    win[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1] = True
    out = np.empty_like(np.asarray(pose_shifts, dtype=float))
    for k in range(n):
        sel = win if valid is None else win & valid[k]

        def predict(p: NDArray[np.float64]) -> NDArray[np.float64]:
            pred = np.zeros((ny, nx))
            for i, b in enumerate(bands):
                img = fourier_shift(models[b], (upfact * p[0], upfact * p[1]))[::upfact, ::upfact]
                pred = np.where(band_map == i, p[2 + i] * img, pred)
            return pred

        p0 = np.r_[pose_shifts[k], np.ones(len(bands))]
        res = least_squares(lambda p: (frames[k] - predict(p))[sel], p0,
                            x_scale=np.r_[0.1, 0.1, 0.1 * np.ones(len(bands))])
        out[k] = res.x[:2]
    return out - out[0]


@dataclass
class BayerResult:
    """Résultat d'une reconstruction par bande.

    Attributes
    ----------
    bands : dict of str to SpriteResult
        Reconstruction de chaque bande (``"R"``, ``"G"``, ``"B"``), sur une
        grille HR commune de pas ``1 / upfact`` pixel du détecteur.
    pose_shifts : ndarray, shape (n_poses, 2)
        Décalages achromatiques entre poses, en pixels du détecteur.
    chromatic : dict
        Pour chaque bande : centroïde ``(y, x)`` en pixels HR, FWHM en pixels
        HR, ellipticités ; et les décalages latéraux R − G et B − G.
    """

    bands: dict[str, SpriteResult]
    pose_shifts: NDArray[np.float64]
    chromatic: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        """Résumé chromatique lisible.

        Returns
        -------
        str
            Une ligne par bande, puis les décalages latéraux.
        """
        lines = ["Bande  centroïde (y, x) HR   FWHM HR    e1       e2"]
        for b in ("R", "G", "B"):
            c = self.chromatic[b]
            lines.append(f"  {b}    ({c['centroid'][0]:7.3f}, {c['centroid'][1]:7.3f})  {c['fwhm']:7.3f}  "
                         f"{c['e1']:+.4f}  {c['e2']:+.4f}")
        for k in ("R-G", "B-G"):
            d = self.chromatic[k]
            lines.append(f"Décalage {k} : ({d[0]:+.3f}, {d[1]:+.3f}) pixel HR")
        return "\n".join(lines)


def _centroid(img: NDArray[np.float64]) -> NDArray[np.float64]:
    from .estimation import weighted_centroid

    return weighted_centroid(img / img.sum(), sigma=4.0, n_iter=10) - 1.0


def band_problems(
    raw: NDArray[np.float64], cfg: SpriteConfig, pattern: str, pose_shifts: NDArray[np.float64]
) -> dict[str, tuple[NDArray[np.float64], SpriteConfig]]:
    """Problème SPRITE monochrome de chaque bande : images des sous-réseaux et paramètres.

    Les images d'une bande sont rangées par sous-réseau : pour G (deux
    sous-réseaux), l'image ``s * n_poses + k`` est le sous-réseau ``s`` de la
    pose ``k``. Les décalages des sous-réseaux sont imposés (:func:`lattice_shifts`).

    Parameters
    ----------
    raw : ndarray, shape (n_poses, ny, nx)
        Mosaïques brutes.
    cfg : SpriteConfig
        Paramètres communs ; ``upfact`` par rapport au pixel du détecteur.
    pattern : str
        Motif de Bayer.
    pose_shifts : ndarray, shape (n_poses, 2)
        Décalages entre poses, en pixels du détecteur.

    Returns
    -------
    dict of str to tuple
        Pour chaque bande, ``(images, paramètres)`` à passer à :func:`sprite.solver.sprite`.
    """
    bad = None if cfg.bad_pixels is None else np.broadcast_to(np.asarray(cfg.bad_pixels, bool), raw.shape)
    out: dict[str, tuple[NDArray[np.float64], SpriteConfig]] = {}
    for band, subs in split_bayer(raw, pattern).items():
        band_frames = np.concatenate([sub for sub, _ in subs])
        band_shifts = np.concatenate([lattice_shifts(pose_shifts, off) for _, off in subs])
        band_bad = None if bad is None else np.concatenate([bad[:, oy::2, ox::2] for _, (oy, ox) in subs])
        fluxes = cfg.fluxes
        if fluxes is not None:
            fluxes = np.tile(np.asarray(fluxes, dtype=float), len(subs))
        out[band] = (band_frames, replace(cfg, upfact=2 * cfg.upfact, shifts=band_shifts, bad_pixels=band_bad,
                                          fluxes=fluxes, ref_frame=0))
    return out


def sprite_bayer(
    frames: ArrayLike,
    config: SpriteConfig | None = None,
    pattern: str = "RGGB",
    pose_shifts: ArrayLike | None = None,
    pose_refine: int | None = None,
    n_jobs: int | None = None,
    **kwargs: Any,
) -> BayerResult:
    """Reconstruit une PSF HR par bande à partir de poses brutes en mosaïque de Bayer.

    Parameters
    ----------
    frames : array_like, shape (n_poses, ny, nx)
        Images brutes non dématricées, de taille paire.
    config : SpriteConfig, optional
        Paramètres communs aux bandes. ``upfact`` est le facteur de
        sur-échantillonnage par rapport au pixel du détecteur (2 par défaut) ;
        chaque sous-réseau est reconstruit avec ``2 upfact``. ``bad_pixels``
        (carte du détecteur) et ``saturation`` s'appliquent à la mosaïque ;
        ``shifts`` est ignoré au profit de ``pose_shifts``.
    pattern : str, default "RGGB"
        Motif de Bayer.
    pose_shifts : array_like, shape (n_poses, 2), optional
        Décalages connus entre poses, en pixels du détecteur ; estimés sinon.
    pose_refine : int, optional
        Nombre de réajustements des décalages sur le modèle reconstruit (voir
        :func:`refine_pose_shifts`), chacun suivi d'une nouvelle
        reconstruction ; sans effet si ``pose_shifts`` est fourni. Par défaut,
        2 à partir de 50 poses, 0 en dessous : sur données simulées, il réduit
        l'erreur des décalages d'environ un tiers avec 100 poses, mais son effet
        est irrégulier avec 16 poses (dégénérescence entre PSF et décalages).
    n_jobs : int, optional
        Nombre de processus : les trois bandes sont reconstruites en parallèle
        (voir :func:`sprite.parallel.run_parallel`).
    **kwargs
        Remplacent les champs de ``config``.

    Returns
    -------
    BayerResult
        PSF de chaque bande et diagnostic chromatique.
    """
    cfg = replace(config or SpriteConfig(), **kwargs)
    raw = np.asarray(frames, dtype=float)
    n = raw.shape[0]
    up = cfg.upfact

    est_frames = raw
    if cfg.subtract_background:
        # Fond par sous-réseau, pour l'estimation des décalages seulement ;
        # chaque reconstruction de bande soustrait le sien.
        from .io import border_background

        est_frames = raw.copy()
        for band in split_bayer(raw, pattern).values():
            for sub, (oy, ox) in band:
                est_frames[:, oy::2, ox::2] -= border_background(sub)[:, None, None]
    shifts = (np.asarray(pose_shifts, dtype=float) if pose_shifts is not None
              else estimate_pose_shifts(est_frames, pattern, cfg))

    bad = None if cfg.bad_pixels is None else np.broadcast_to(np.asarray(cfg.bad_pixels, bool), raw.shape)

    def reconstruct(shifts: NDArray[np.float64]) -> dict[str, SpriteResult]:
        problems = band_problems(raw, cfg, pattern, shifts)
        solved = run_parallel(sprite, list(problems.values()), n_jobs)
        return dict(zip(problems, solved))

    results = reconstruct(shifts)
    if pose_refine is None:
        pose_refine = 2 if n >= 50 else 0
    n_refine = 0 if pose_shifts is not None else pose_refine
    valid = None if bad is None else ~bad
    for _ in range(n_refine):
        models = {b: r.solution for b, r in results.items()}
        shifts = refine_pose_shifts(est_frames, models, shifts, up, pattern, valid)
        results = reconstruct(shifts)

    chromatic: dict[str, Any] = {}
    for band, res in results.items():
        sol = res.solution / res.solution.sum()
        e1, e2 = ellipticity(sol)
        chromatic[band] = {"centroid": _centroid(sol), "fwhm": fwhm(sol), "e1": e1, "e2": e2}
    chromatic["R-G"] = chromatic["R"]["centroid"] - chromatic["G"]["centroid"]
    chromatic["B-G"] = chromatic["B"]["centroid"] - chromatic["G"]["centroid"]
    return BayerResult(bands=results, pose_shifts=shifts, chromatic=chromatic)
