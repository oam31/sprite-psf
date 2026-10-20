"""Calibration de sprite-psf-check sur données simulées (vérité connue).

Pour chaque cas, compare les indicateurs du contrôle (chi2 d'ajustement, chi2 de
prédiction croisée, écart entre moitiés) à l'erreur vraie de la reconstruction.

Usage : python validation/check_calibration.py
"""

from __future__ import annotations

import numpy as np

from flux_benchmark import simulate
from sprite.diagnostics import check
from sprite.metrics import compare

CASES = [
    ("correct, 16 images, 30 dB", dict(n=16, snr=30, spread=0.0), dict()),
    ("correct, 16 images, 20 dB", dict(n=16, snr=20, spread=0.0), dict()),
    ("correct, 64 images, 30 dB", dict(n=64, snr=30, spread=0.0), dict()),
    ("décalages du C++ (centroid)", dict(n=16, snr=30, spread=0.0), dict(shift_method="centroid")),
    ("flux ±20 % non estimés", dict(n=16, snr=30, spread=0.2), dict()),
    ("flux ±20 % estimés (fit + 3)", dict(n=16, snr=30, spread=0.2),
     dict(estimate_flux=True, flux_method="fit", flux_refine=3)),
    ("régularisation faible (kappa 1)", dict(n=16, snr=30, spread=0.0), dict(kappa=1.0)),
    ("régularisation forte (kappa 8)", dict(n=16, snr=30, spread=0.0), dict(kappa=8.0)),
]


def main() -> None:
    print("starlet, -N, kappa 4 sauf mention ; 2 réalisations par cas")
    print(f"{'cas':<34}{'chi2 fit':>9}{'chi2 préd.':>11}{'rapport':>8}{'moitiés rel':>12}{'moitiés FWHM':>13}"
          f"{'disp. flux':>11}{'erreur vraie rel':>17}{'FWHM vraie':>11}")
    for name, sim, opts in CASES:
        rows = []
        for r in range(2):
            rng = np.random.default_rng(4000 + 10 * r + sim["n"])
            truth, frames, _ = simulate(sim["snr"], rng, sim["n"], sim["spread"])
            kw = dict(transform="starlet", kappa=4, estimate_noise_per_frame=True)
            kw.update(opts)
            res, rep = check(frames, **kw)
            err = compare(res.solution, truth)
            rows.append([np.median(rep.chi2), np.median(rep.holdout_chi2),
                         np.median(rep.holdout_chi2) / np.median(rep.chi2),
                         rep.split["rel_err"], rep.split["fwhm_err"], rep.flux_dispersion,
                         err["rel_err"], err["fwhm_err"]])
        m = np.mean(rows, axis=0)
        print(f"{name:<34}{m[0]:>9.3f}{m[1]:>11.3f}{m[2]:>8.3f}{m[3]:>12.4f}{100 * m[4]:>12.2f}%"
              f"{100 * m[5]:>10.1f}%{m[6]:>17.4f}{100 * m[7]:>10.2f}%", flush=True)


if __name__ == "__main__":
    main()
