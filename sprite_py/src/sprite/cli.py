"""Interfaces en ligne de commande : ``sprite-psf`` (reconstruction) et ``sprite-psf-check`` (contrôle).

Exemples ::

    # données réelles : TIFF 16 bits, une image par pose, réglages conseillés
    sprite-psf -t 2 -s 4 -N --background --saturation 65000 --bad-pixels defauts.tif \\
        images/*.tif psf.tif sorties/

    # contrôle de la même reconstruction (ajustement, prédiction croisée, moitiés)
    sprite-psf-check -t 2 -s 4 -N --background --saturation 65000 --bad-pixels defauts.tif \\
        images/*.tif

    # configuration du papier, reproduite à l'identique (erreurs du C++ comprises)
    sprite-psf -t 2 -s 4 -F -N --cpp-compat cube_test.fits solution.fits sorties/

Entrées : cube FITS, TIFF multipage ou multibande, ou plusieurs fichiers d'une
image chacun, lus dans l'ordre donné. Le format de sortie suit l'extension du
fichier (``.fits`` ou ``.tif``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .io import load_bad_pixels, load_frames, save_image
from .masking import summary
from .shifts import METHODS
from .solver import SpriteConfig, sprite


def _add_config_arguments(p: argparse.ArgumentParser) -> None:
    """Ajoute les options de reconstruction, regroupées par thème."""
    main = p.add_argument_group("paramètres principaux")
    main.add_argument("-t", dest="transform", default="2",
                      help="dictionnaire : 2 (starlet, défaut) ou 24 (7/9 non décimée, défaut de run_sprite)")
    main.add_argument("-s", dest="kappa", type=float, default=4.0,
                      help="seuil de parcimonie kappa, en écarts types (défaut 4 ; 3 dans run_sprite)")
    main.add_argument("-N", dest="estimate_noise_per_frame", action="store_true",
                      help="bruit estimé image par image (conseillé)")
    main.add_argument("-r", dest="upfact", type=int, default=2, help="facteur de sur-échantillonnage (défaut 2)")

    shifts = p.add_argument_group("décalages")
    shifts.add_argument("--shift-method", default=None, choices=METHODS,
                        help="estimateur (défaut : centroid_wide)")
    shifts.add_argument("--wide-factor", type=float, default=2.5,
                        help="fenêtre de centroid_wide, en largeurs de PSF (défaut 2,5)")
    shifts.add_argument("-S", dest="shift_file", help="décalages connus : une ligne 'sx sy' par image, pixels BR")
    shifts.add_argument("--shift-refine", type=int, default=0,
                        help="réajustements des décalages sur la solution (défaut 0, déconseillé)")

    flux = p.add_argument_group("flux")
    flux.add_argument("-F", dest="estimate_flux", action="store_true",
                      help="estime les flux relatifs (seulement s'ils varient entre poses)")
    flux.add_argument("--flux-method", choices=["aperture", "fit"], default="aperture",
                      help="avec -F : ouverture (méthode du papier) ou ajustement sur un modèle (conseillé)")
    flux.add_argument("--flux-refine", type=int, default=0,
                      help="avec -F : réajustements des flux sur la solution (3 conseillés)")
    flux.add_argument("--fluxes", help="flux relatifs connus : une valeur par image (prioritaire sur -F)")

    pix = p.add_argument_group("fond et pixels")
    pix.add_argument("--background", action="store_true",
                     help="soustrait la médiane de la bordure de chaque image (offset du détecteur)")
    pix.add_argument("--saturation", type=float, default=None,
                     help="niveau de saturation (données brutes) : pixels traités comme bornes inférieures")
    pix.add_argument("--bad-pixels", nargs="+", default=None,
                     help="carte(s) de pixels défectueux, FITS ou TIFF, non nul = défectueux")
    pix.add_argument("--mask-dilate", type=int, default=0,
                     help="couronne exclue autour des zones saturées ou défectueuses, en pixels (défaut 0)")

    algo = p.add_argument_group("algorithme (avancé)")
    algo.add_argument("-n", dest="n_scales", type=int, default=-1, help="nombre d'échelles (défaut automatique)")
    algo.add_argument("-i", dest="n_iter", type=int, default=50, help="itérations maximales par passe (défaut 50)")
    algo.add_argument("-R", dest="n_reweight", type=int, default=1, help="passes de repondération (défaut 1)")
    algo.add_argument("-C", dest="cv_win", type=int, default=25, help="fenêtre du critère d'arrêt (défaut 25)")
    algo.add_argument("-a", dest="cv_tol", type=float, default=0.005, help="tolérance du critère d'arrêt")
    algo.add_argument("--prox-backprojection", choices=["auto", "adjoint", "recons"], default="auto",
                      help="prox d'analyse : adjoint exact ou reconstruction (auto : adjoint pour la starlet, "
                           "reconstruction pour la 7/9)")

    compat = p.add_argument_group(
        "rétrocompatibilité avec le C++ publié",
        "Ces options reproduisent des erreurs du code d'origine, pour retrouver ses résultats à l'identique. "
        "Elles ne doivent pas être utilisées pour des mesures.")
    compat.add_argument("--cpp-compat", action="store_true",
                        help="reproduit les quatre erreurs du C++ et son estimateur de décalages centroid")
    compat.add_argument("--cpp-compat-stack", action="store_true",
                        help="seulement l'erreur de recalage de la première estimation")
    compat.add_argument("--cpp-compat-flux", action="store_true",
                        help="seulement l'erreur d'ouverture de flux (décalée d'un pixel)")
    compat.add_argument("--cpp-compat-stopping", action="store_true",
                        help="seulement l'erreur du critère d'arrêt")


def config_from_args(args: argparse.Namespace) -> SpriteConfig:
    """Construit la configuration à partir des options analysées.

    Parameters
    ----------
    args : argparse.Namespace
        Options de :func:`build_parser` ou :func:`build_check_parser`.

    Returns
    -------
    SpriteConfig
        Configuration correspondante.
    """
    shifts = None
    if args.shift_file:
        sx, sy = np.loadtxt(args.shift_file, ndmin=2).T  # ordre du C++ : x puis y
        shifts = np.c_[sy, sx]
    return SpriteConfig(
        upfact=args.upfact, transform=args.transform, n_scales=args.n_scales, kappa=args.kappa,
        n_iter=args.n_iter, n_reweight=args.n_reweight, cv_win=args.cv_win, cv_tol=args.cv_tol,
        estimate_flux=args.estimate_flux, estimate_noise_per_frame=args.estimate_noise_per_frame,
        flux_method=args.flux_method, flux_refine=args.flux_refine,
        fluxes=np.loadtxt(args.fluxes, ndmin=1) if args.fluxes else None,
        shifts=shifts, shift_method=args.shift_method, wide_factor=args.wide_factor,
        shift_refine=args.shift_refine, subtract_background=args.background,
        saturation=args.saturation,
        bad_pixels=load_bad_pixels(args.bad_pixels) if args.bad_pixels else None,
        mask_dilate=args.mask_dilate,
        exact_adjoint={"auto": None, "adjoint": True, "recons": False}[args.prox_backprojection],
        cpp_compat=args.cpp_compat,
        compat_stack=True if args.cpp_compat_stack else None,
        compat_flux=True if args.cpp_compat_flux else None,
        compat_stopping=True if args.cpp_compat_stopping else None,
        verbose=args.verbose,
    )


def _add_bayer_arguments(p: argparse.ArgumentParser) -> None:
    """Options de la mosaïque de Bayer et du parallélisme, communes aux deux commandes."""
    bay = p.add_argument_group("mosaïque de Bayer")
    bay.add_argument("--bayer", choices=["RGGB", "BGGR", "GRBG", "GBRG"], default=None,
                     help="images brutes en mosaïque (non dématricées) : une PSF par bande R, G, B, "
                          "sur une grille commune ; -r est le facteur par rapport au pixel du détecteur")
    p.add_argument("-j", "--jobs", type=int, default=None,
                   help="nombre de processus pour les reconstructions indépendantes (bandes, blocs, "
                        "moitiés, valeurs balayées) ; défaut : nombre de cœurs moins un")


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur des arguments de ``sprite-psf``.

    Returns
    -------
    argparse.ArgumentParser
        Analyseur configuré.
    """
    p = argparse.ArgumentParser(prog="sprite-psf", description="Super-résolution de PSF par SPRITE")
    p.add_argument("inputs", nargs="+",
                   help="images BR : cube FITS, TIFF multipage ou multibande, plusieurs fichiers ou un dossier")
    p.add_argument("output", help="fichier de sortie (.fits ou .tif)")
    p.add_argument("outdir", help="dossier de sortie")
    _add_config_arguments(p)
    _add_bayer_arguments(p)
    out = p.add_argument_group("sorties")
    out.add_argument("-w", dest="resi_file", help="écrit le cube des résidus BR dans ce fichier")
    out.add_argument("-W", dest="diagnostics", action="store_true",
                     help="écrit les fichiers de diagnostic (flux.txt, shift.txt, pixel_map...)")
    out.add_argument("-v", dest="verbose", action="store_true", help="affiche les étapes")
    out.add_argument("--cpp-orientation", action="store_true",
                     help="écrit les images 2D transposées, comme le C++ (rétrocompatibilité)")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée de ``sprite-psf``.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments ; ceux de la ligne de commande par défaut.
    """
    args = build_parser().parse_args(argv)
    frames, dtypes = load_frames(args.inputs)
    if args.verbose:
        print(f"{len(frames)} images de {frames.shape[1]} x {frames.shape[2]} ({', '.join(sorted(set(dtypes)))})")
    cfg = config_from_args(args)
    if args.bayer:
        _main_bayer(args, frames, cfg)
        return
    res = sprite(frames, cfg)
    if args.verbose and not res.valid.all():
        print(summary(res.valid, res.censored))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(args.output).suffix or ".fits"
    img2d = (lambda a: a.T) if args.cpp_orientation else (lambda a: a)
    save_image(out / args.output, img2d(res.solution))
    save_image(out / f"lr_{Path(args.output).stem}{ext}", res.lr_model)
    if args.resi_file:
        save_image(out / args.resi_file, res.residual)
    if args.diagnostics:
        save_image(out / f"model{ext}", img2d(res.first_guess))
        # Carte des pixels : 1 valide, 2 saturé (censuré), 0 exclu.
        save_image(out / f"pixel_map{ext}", (res.valid + 2 * res.censored).astype(np.float32))
        # Fichiers texte au format du C++ : colonnes (x, y).
        np.savetxt(out / "centroids.txt", res.centroids[:, ::-1], fmt="%f")
        np.savetxt(out / "shift.txt", res.shifts[:, ::-1], fmt="%f")
        np.savetxt(out / "flux.txt", res.flux, fmt="%f")
        np.savetxt(out / "sig_est.txt", res.sigma, fmt="%f")
        np.savetxt(out / "mse.txt", np.atleast_2d(res.residual_energy), fmt="%f")
        if res.background is not None:
            np.savetxt(out / "background.txt", res.background, fmt="%f")
        (out / "param.txt").write_text(
            f"Estimated spectral radius: {res.spectral_radius:f}\n"
            f"nsigma: {cfg.kappa:f}\n"
            f"Type of transform: {cfg.transform}\n"
            f"Shift method: {cfg.resolved_shift_method()} (refinements: {cfg.shift_refine})\n"
            f"Analysis prox: {'exact adjoint' if cfg.resolved_exact_adjoint() else 'reconstruction'}\n"
            f"PSF size (LR pixels): {res.psf_sigma:f}\n"
            f"Saturated pixels: {int(res.censored.sum())}\n"
            f"Excluded pixels: {int((~res.valid & ~res.censored).sum())}\n"
            f"C++ compatibility: {cfg.cpp_compat}\n"
            f"Number of iterations for convergence: {res.n_iter}\n"
        )
    if args.verbose:
        print(f"Solution écrite dans {out / args.output}")


def _main_bayer(args: argparse.Namespace, frames: np.ndarray, cfg: SpriteConfig) -> None:
    """Reconstruction par bande d'images en mosaïque de Bayer et écriture des sorties."""
    from .bayer import sprite_bayer

    # -S : décalages connus des poses, en pixels du détecteur (colonnes x puis y).
    res = sprite_bayer(frames, cfg, pattern=args.bayer, pose_shifts=cfg.shifts, n_jobs=args.jobs)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem, ext = Path(args.output).stem, Path(args.output).suffix or ".fits"
    for band, r in res.bands.items():
        save_image(out / f"{stem}_{band}{ext}", r.solution)
        if args.diagnostics:
            np.savetxt(out / f"flux_{band}.txt", r.flux, fmt="%f")
            np.savetxt(out / f"sig_est_{band}.txt", r.sigma, fmt="%f")
    report = res.text()
    (out / "chromatique.txt").write_text(report + "\n")
    if args.diagnostics:
        np.savetxt(out / "pose_shift.txt", res.pose_shifts[:, ::-1], fmt="%f")  # pixels détecteur, x puis y
    if args.verbose:
        print(report)
        print(f"PSF écrites dans {out}")


def build_check_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur des arguments de ``sprite-psf-check``.

    Returns
    -------
    argparse.ArgumentParser
        Analyseur configuré.
    """
    p = argparse.ArgumentParser(
        prog="sprite-psf-check",
        description="Contrôle d'une reconstruction SPRITE sur données réelles : ajustement, "
                    "prédiction d'images mises de côté, cohérence entre moitiés (au moins 8 images)")
    p.add_argument("inputs", nargs="+", help="images BR, comme pour sprite-psf")
    _add_config_arguments(p)
    _add_bayer_arguments(p)
    p.add_argument("--folds", type=int, default=5, help="nombre de blocs de la prédiction croisée (défaut 5)")
    p.add_argument("--scan", nargs="+", metavar=("PARAM", "VALEUR"),
                   help="balaye un paramètre (kappa, transform, wide_factor, n_reweight, mask_dilate) "
                        "et indique la valeur retenue par validation croisée ; ex. --scan kappa 2 3 4 5 6 8")
    p.add_argument("--json", help="écrit aussi les résultats dans ce fichier JSON")
    p.add_argument("-v", dest="verbose", action="store_true", help="affiche les étapes")
    return p


def check_main(argv: Sequence[str] | None = None) -> None:
    """Point d'entrée de ``sprite-psf-check``.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments ; ceux de la ligne de commande par défaut.
    """
    from .diagnostics import check

    args = build_check_parser().parse_args(argv)
    frames, _ = load_frames(args.inputs)
    if args.scan:
        from .diagnostics import scan

        if len(args.scan) < 3:
            raise SystemExit("--scan attend un paramètre et au moins deux valeurs")
        if args.bayer:
            raise SystemExit("--scan ne traite pas encore les mosaïques de Bayer : balayer sur la bande G "
                             "n'est pas équivalent ; comparer plutôt des contrôles --bayer successifs")
        param, values = args.scan[0], args.scan[1:]
        rows = scan(frames, config_from_args(args), param, values, n_folds=args.folds, n_jobs=args.jobs)
        print(f"Balayage de {param} : chi2/N de prédiction (± erreur type) ; valeur retenue = la plus "
              "grande à moins d'une erreur type du minimum")
        print(f"{'valeur':>10}{'chi2 ajust.':>13}{'chi2 préd.':>12}{'± e.t.':>9}{'rapport':>9}")
        for r in rows:
            mark = ("  <- retenue" if r["recommended"] else "") + ("  (minimum)" if r["minimum"] else "")
            print(f"{str(r['value']):>10}{r['chi2']:>13.4f}{r['holdout_chi2']:>12.4f}{r['holdout_se']:>9.4f}"
                  f"{r['ratio']:>9.3f}{mark}")
        if args.json:
            Path(args.json).write_text(json.dumps({"scan": param, "rows": rows}, indent=1, ensure_ascii=False))
        return
    cfg = config_from_args(args)
    if args.bayer:
        from .diagnostics import check_bayer

        # -S : décalages connus des poses, en pixels du détecteur.
        _, breport = check_bayer(frames, replace(cfg, shifts=None), pattern=args.bayer, pose_shifts=cfg.shifts,
                                 n_folds=args.folds, n_jobs=args.jobs)
        print(breport.text())
        if args.json:
            Path(args.json).write_text(json.dumps({
                "bayer": args.bayer, "pose_shifts": breport.pose_shifts.tolist(),
                "chromatic": {k: v.tolist() for k, v in breport.chromatic.items()},
                "chromatic_split": breport.chromatic_split,
                "bands": {b: _report_dict(r) for b, r in breport.bands.items()},
            }, indent=1, ensure_ascii=False))
        return
    _, report = check(frames, cfg, n_folds=args.folds, n_jobs=args.jobs)
    print(report.text())
    if args.json:
        Path(args.json).write_text(json.dumps(_report_dict(report), indent=1, ensure_ascii=False))


def _report_dict(report: Any) -> dict[str, Any]:
    """Contenu JSON d'un :class:`sprite.diagnostics.CheckReport`."""
    return {
        "chi2": report.chi2.tolist(), "holdout_chi2": report.holdout_chi2.tolist(),
        "split": report.split, "shape": report.shape, "flux": report.flux.tolist(),
        "flux_dispersion": report.flux_dispersion, "shifts_hr": report.shifts.tolist(),
        "notes": report.notes,
    }


if __name__ == "__main__":
    main()
