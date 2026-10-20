"""Estimation des décalages entre poses en mosaïque de Bayer : cas de 100 poses.

Cas visé : décalages inconnus, de plusieurs pixels au total sur la série ;
PSF rouge la plus dégradée (aberrations supplémentaires sur le chemin rouge).
Compare l'estimation directe (pseudo-luminance) et le réajustement sur les PSF
reconstruites, en erreur sur les décalages et en erreur des PSF par bande.

Usage : python validation/bayer_shifts.py
"""

from __future__ import annotations

import time

import numpy as np

import bayer_sim
from sprite import SpriteConfig
from sprite.bayer import sprite_bayer
from sprite.metrics import compare


def main() -> None:
    # PSF rouge plus dégradée : défocalisation de 0,3 onde (à lambda_G) au lieu de 0,1.
    bayer_sim.DEFOCUS["R"] = 0.30
    cfg = SpriteConfig(estimate_noise_per_frame=True)
    print("100 poses, décalages uniformes ±2,5 pixels, SNR 30 dB, R défocalisé de 0,3 onde ; "
          "erreur relative / FWHM par bande")
    print(f"{'variante':<28}{'err. décal.':>12}{'R':>17}{'G':>17}{'B':>17}")
    t0 = time.time()
    for r in range(2):
        truth, mosaic, _, pose = bayer_sim.simulate(500 + r, 100, 2.5, 30)
        for label, kw in (("décalages vrais", dict(pose_shifts=pose)),
                          ("estimés (luminance)", dict(pose_refine=0)),
                          ("estimés + 1 réajustement", dict(pose_refine=1)),
                          ("estimés + 2 réajustements", dict(pose_refine=2))):
            res = sprite_bayer(mosaic, cfg, **kw)
            err = float(np.sqrt(np.mean((res.pose_shifts[1:] - pose[1:]) ** 2)))
            cells = "".join(
                f"{compare(res.bands[b].solution, truth[b])['rel_err']:>9.3f} / "
                f"{100 * compare(res.bands[b].solution, truth[b])['fwhm_err']:3.1f}%" for b in "RGB")
            print(f"r{r} {label:<25}{err:>9.3f} px{cells}   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
