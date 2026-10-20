"""Calibration de sprite-psf-check --bayer sur mosaïques simulées (vérité connue).

Pour chaque cas, compare les indicateurs du contrôle de chaque bande (chi2
d'ajustement, chi2 de prédiction croisée par poses, écart entre moitiés,
dispersion des flux, stabilité des décalages chromatiques) à l'erreur vraie.
Simulation de validation/bayer_sim.py : RGGB, poses à ±1 pixel, 30 dB.

Usage : python validation/bayer_check_calibration.py [--jobs N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bayer_sim import simulate
from sprite.bayer import _centroid
from sprite.diagnostics import check_bayer
from sprite.metrics import compare

CASES = [
    ("50 poses, décalages connus", dict(n=50), dict(known=True)),
    ("50 poses, décalages estimés", dict(n=50), dict()),
    ("100 poses, décalages estimés", dict(n=100), dict()),
    ("16 poses, décalages connus", dict(n=16), dict(known=True)),
    ("16 poses, décalages estimés", dict(n=16), dict()),
    ("50 poses, sans réajustement", dict(n=50), dict(pose_refine=0)),
    ("50 poses, flux ±20 % non estimés", dict(n=50, spread=0.2), dict()),
    ("50 poses, kappa 1", dict(n=50), dict(kappa=1.0)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    print("RGGB, poses ±1 pixel, 30 dB, starlet, kappa 4 sauf mention, -N ; une réalisation par cas")
    print("chromatique : écart entre moitiés | erreur vraie, norme des décalages R-G et B-G (pixels HR)")
    print(f"{'cas':<34}{'b':>2}{'fit':>7}{'préd.':>7}{'rapp.':>7}{'moit. rel':>10}{'moit. FWHM':>11}"
          f"{'disp.':>7}{'vrai rel':>9}{'vrai FWHM':>10}")
    for name, sim, opts in CASES:
        truth, mosaic, _, pose = simulate(0, sim["n"], 1.0, 30)
        spread = sim.get("spread", 0.0)
        if spread:
            flux = np.random.default_rng(7).uniform(1 - spread, 1 + spread, sim["n"])
            flux[0] = 1.0
            mosaic = mosaic * flux[:, None, None]
        kw = dict(estimate_noise_per_frame=True)
        kw.update({k: v for k, v in opts.items() if k != "known"})
        t = time.time()
        full, rep = check_bayer(mosaic, pose_shifts=pose if opts.get("known") else None, n_jobs=args.jobs, **kw)
        elapsed = time.time() - t
        for i, (b, r) in enumerate(rep.bands.items()):
            c = compare(full.bands[b].solution, truth[b])
            fit, hold = np.median(r.chi2), np.nanmedian(r.holdout_chi2)
            label = name if i == 0 else ""
            print(f"{label:<34}{b:>2}{fit:>7.3f}{hold:>7.3f}{hold / fit:>7.3f}{r.split['rel_err']:>10.3f}"
                  f"{100 * r.split['fwhm_err']:>10.1f}%{100 * r.flux_dispersion:>6.1f}%{c['rel_err']:>9.3f}"
                  f"{100 * c['fwhm_err']:>9.1f}%")
        true_c = {b: _centroid(truth[b] / truth[b].sum()) for b in truth}
        chrom = []
        for k, b in (("R-G", "R"), ("B-G", "B")):
            err = np.linalg.norm(rep.chromatic[k] - (true_c[b] - true_c["G"]))
            chrom.append(f"{k} {rep.chromatic_split[k]:.3f} | {err:.3f}")
        shift_err = np.sqrt(np.mean((full.pose_shifts - pose) ** 2))
        print(f"{'':<34}   chromatique {', '.join(chrom)} ; erreur décalages {shift_err:.3f} px ; "
              f"{elapsed:.0f} s ; alertes : {len(rep.notes)}")


if __name__ == "__main__":
    main()
