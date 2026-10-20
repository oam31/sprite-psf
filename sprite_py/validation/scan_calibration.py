"""Le minimum du chi2 de prédiction désigne-t-il le meilleur kappa ? (données simulées)

16 images, starlet, -N ; pour chaque kappa : chi2 de prédiction (sprite-psf-check
--scan) et erreur vraie de la reconstruction complète.
"""

from __future__ import annotations

import numpy as np

from flux_benchmark import simulate
from sprite import SpriteConfig, sprite
from sprite.diagnostics import scan
from sprite.metrics import compare

KAPPAS = [2, 3, 4, 5, 6, 8]


def main() -> None:
    print("16 images, starlet, -N ; chi2 de prédiction médian / erreur relative vraie / erreur de FWHM vraie")
    for snr in [20, 30]:
        for r in range(2):
            rng = np.random.default_rng(6000 + 10 * snr + r)
            truth, frames, _ = simulate(snr, rng, 16, 0.0)
            cfg = SpriteConfig(estimate_noise_per_frame=True)
            rows = scan(frames, cfg, "kappa", KAPPAS)
            errs = [compare(sprite(frames, cfg, kappa=k).solution, truth) for k in KAPPAS]
            k_pred = next(x["value"] for x in rows if x["recommended"])
            k_min = next(x["value"] for x in rows if x["minimum"])
            k_true = KAPPAS[int(np.argmin([e["rel_err"] for e in errs]))]
            cells = "  ".join(f"k={k}: {x['holdout_chi2']:.4f}/{e['rel_err']:.3f}/{100 * e['fwhm_err']:.1f}%"
                              for k, x, e in zip(KAPPAS, rows, errs))
            e_pred = errs[KAPPAS.index(k_pred)]["rel_err"]
            e_best = min(e["rel_err"] for e in errs)
            print(f"SNR {snr} r{r}: minimum {k_min}, retenu {k_pred} (erreur {e_pred:.3f}), "
                  f"meilleur vrai {k_true} (erreur {e_best:.3f}) | {cells}", flush=True)


if __name__ == "__main__":
    main()
