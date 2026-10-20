"""Opportunité de l'estimation des flux, sur données synthétiques (voir synthetic.py).

Images d'une même PSF, de flux égaux ou dispersés (uniformes dans [1 - d, 1 + d]),
bruit blanc de même écart type pour toutes. Variantes comparées : flux supposés
égaux, ouverture (C++ corrigé), ajustement sur la première estimation, ajustement
puis un réajustement sur la solution, flux vrais.

Usage : python validation/flux_benchmark.py [--n-real 3] [--out fichier.json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sprite import sprite
from synthetic import HR, Q_HR, hr_psf, metrics, pupil

VARIANTS = {
    "flux égaux": dict(estimate_flux=False),
    "ouverture": dict(estimate_flux=True, flux_method="aperture"),
    "ajustement": dict(estimate_flux=True, flux_method="fit"),
    "ajustement + 1 réaj.": dict(estimate_flux=True, flux_method="fit", flux_refine=1),
    "flux vrais": dict(true_flux=True),
}


def simulate(snr_db, rng, n_frames, spread):
    amp, phase = pupil(HR, HR / Q_HR, rng=rng)
    truth = hr_psf(amp, phase)
    shifts = np.vstack([[0.0, 0.0], rng.uniform(-1, 1, size=(n_frames - 1, 2))])
    flux = np.r_[1.0, rng.uniform(1 - spread, 1 + spread, n_frames - 1)]
    frames = np.array([f * hr_psf(amp, phase, s)[::2, ::2] for f, s in zip(flux, shifts)])
    c = HR // 2
    sigma = np.sqrt(truth[c - 25:c + 25, c - 25:c + 25].var() / 10 ** (snr_db / 10))
    return truth, frames + rng.normal(scale=sigma, size=frames.shape), flux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 30])
    ap.add_argument("--n-frames", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--spread", type=float, nargs="+", default=[0.0, 0.2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for n in args.n_frames:
        for spread in args.spread:
            for snr in args.snr:
                acc = {v: [] for v in VARIANTS}
                for r in range(args.n_real):
                    rng = np.random.default_rng(int(10000 * spread) + 100 * n + int(snr) + 7 * r)
                    truth, frames, flux = simulate(snr, rng, n, spread)
                    for name, opts in VARIANTS.items():
                        opts = dict(opts)
                        if opts.pop("true_flux", False):
                            opts["fluxes"] = flux
                        res = sprite(frames, kappa=4, estimate_noise_per_frame=True, **opts)
                        m = metrics(res.solution, truth)
                        m["flux_rms"] = float(np.sqrt(np.mean((res.flux - flux) ** 2)))
                        acc[name].append(m)
                for name, ms in acc.items():
                    row = {"n_frames": n, "spread": spread, "snr": snr, "variant": name}
                    row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
                    rows.append(row)
                print(f"n={n} dispersion={spread} SNR={snr} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'n':>3} {'disp.':>6} {'SNR':>4}  {'variante':<22}{'err. flux':>10}{'rel_err':>9}{'fwhm':>8}{'e1':>8}{'e2':>8}")
    for r in rows:
        print(f"{r['n_frames']:>3} {r['spread']:>6.1f} {r['snr']:>4.0f}  {r['variant']:<22}{r['flux_rms']:>10.4f}"
              f"{r['rel_err']:>9.3f}{r['fwhm_err']:>8.3f}{r['e1_err']:>8.4f}{r['e2_err']:>8.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
