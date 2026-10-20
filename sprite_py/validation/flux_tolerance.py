"""Variation de flux tolérable ou rattrapable entre poses (voir flux_benchmark.py).

Flux uniformes dans [1 - d, 1 + d] (écart type d / sqrt(3)), 16 images, starlet,
-s 4 -N. Variantes : flux supposés égaux, ajustement + 3 réajustements, flux vrais.

Usage : python validation/flux_tolerance.py
"""

from __future__ import annotations

import numpy as np

from flux_benchmark import simulate
from sprite import sprite
from synthetic import metrics

SPREADS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
VARIANTS = {
    "flux égaux": dict(estimate_flux=False),
    "ajustement + 3 réaj.": dict(estimate_flux=True, flux_method="fit", flux_refine=3),
    "flux vrais": dict(true_flux=True),
}


def main() -> None:
    print("16 images, starlet, -s 4 -N, 3 réalisations ; erreur relative / erreur de FWHM / erreur RMS des flux")
    print(f"{'SNR':>4} {'d':>5} {'écart type':>11}  " + "".join(f"{v:>30}" for v in VARIANTS))
    for snr in [20, 30]:
        for d in SPREADS:
            cells = []
            for name, opts in VARIANTS.items():
                ms = []
                for r in range(3):
                    rng = np.random.default_rng(int(1000 * d) + 10 * int(snr) + r)
                    truth, frames, flux = simulate(snr, rng, 16, d)
                    o = dict(opts)
                    if o.pop("true_flux", False):
                        o["fluxes"] = flux
                    res = sprite(frames, transform="starlet", kappa=4, estimate_noise_per_frame=True, **o)
                    m = metrics(res.solution, truth)
                    m["flux"] = float(np.sqrt(np.mean((res.flux - flux) ** 2)))
                    ms.append(m)
                cells.append(f"{np.mean([m['rel_err'] for m in ms]):.3f} / {100 * np.mean([m['fwhm_err'] for m in ms]):4.1f} % / {100 * np.mean([m['flux'] for m in ms]):4.1f} %")
            print(f"{snr:>4} {d:>5.2f} {100 * d / np.sqrt(3):>9.1f} %  " + "".join(f"{c:>30}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
