"""Compare the shift-estimation options on synthetic data (see synthetic.py).

For each SNR and realisation, reports the RMS shift error (HR pixels) and the
SPRITE reconstruction metrics obtained with each option. Fluxes are equal in
the simulation, so flux estimation is off.

Usage: python validation/shifts_benchmark.py [--transform u79] [--n-real 4] [--out file.json]
"""

import argparse
import json
import time

import numpy as np

from sprite import sprite
from synthetic import HR, Q_HR, metrics, pupil, simulate

VARIANTS = {
    "centroid": dict(shift_method="centroid"),
    "centroid_wide": dict(shift_method="centroid_wide"),
    "xcorr": dict(shift_method="xcorr"),
    "centroid_wide+refine": dict(shift_method="centroid_wide", shift_refine=1),
    "true_shifts": dict(true_shifts=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="u79")
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = {v: {str(s): [] for s in args.snr} for v in VARIANTS}
    t0 = time.time()
    for snr in args.snr:
        for r in range(args.n_real):
            rng = np.random.default_rng(1000 * int(snr) + r)
            amp, phase = pupil(HR, HR / Q_HR, rng=rng)
            truth, frames, shifts = simulate(snr, rng, amp, phase)
            for name, opts in VARIANTS.items():
                opts = dict(opts)
                if opts.pop("true_shifts", False):
                    opts["shifts"] = shifts / 2
                res = sprite(frames, transform=args.transform, kappa=4,
                             estimate_noise_per_frame=True, **opts)
                row = metrics(res.solution, truth)
                row["shift_rms"] = float(np.sqrt(np.mean((res.shifts[1:] - shifts[1:]) ** 2)))
                raw[name][str(snr)].append(row)
            print(f"SNR {snr:>4} dB  real {r + 1}/{args.n_real}  ({time.time() - t0:.0f}s)", flush=True)

    keys = ["shift_rms", "rel_err", "e1_err", "e2_err", "fwhm_err"]
    print(f"\ntransform={args.transform}  n_real={args.n_real}")
    print(f"{'variant':<22}{'SNR':>5}" + "".join(f"{k:>11}" for k in keys))
    summary = {}
    for name in VARIANTS:
        for snr in args.snr:
            rows = raw[name][str(snr)]
            means = {k: float(np.mean([row[k] for row in rows])) for k in keys}
            summary.setdefault(name, {})[str(snr)] = means
            print(f"{name:<22}{snr:>5.0f}" + "".join(f"{means[k]:>11.4f}" for k in keys))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "raw": raw}, f, indent=1)


if __name__ == "__main__":
    main()
