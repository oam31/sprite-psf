"""Synthetic validation following the protocol of Ngole Mboula et al. (2015), section 4.

A high-resolution (HR) optical PSF is computed from an obscured pupil with a
three-vane spider and low-order aberrations, sampled at 2 x 0.688 x Nyquist.
Four low-resolution (LR) copies are generated with random sub-pixel shifts
(exact Fourier tilt in the pupil, then decimation by 2) plus white Gaussian
noise at a given SNR. Methods are compared with the paper's criteria:
relative error, ellipticity errors |e1|, |e2| and relative FWHM error.

Usage: python validation/synthetic.py [--transform starlet|u79] [--n-real N] [--out file.json]
"""

import argparse
import json
import time

import numpy as np

from sprite import sprite
from sprite.metrics import compare, ellipticity, fwhm
from sprite.first_guess import lanczos_stack

HR = 168          # HR grid (LR frames are 84 x 84 as in cube_test.fits)
Q_HR = 2 * 2 * 0.688  # HR sampling, in units of lambda/D per... samples per lambda/D


def pupil(n, diam, obsc=0.3, spider=0.02, rng=None):
    yy, xx = (np.indices((n, n)) - n / 2) / (diam / 2)
    r = np.hypot(yy, xx)
    p = ((r <= 1) & (r >= obsc)).astype(float)
    for ang in np.deg2rad([90, 210, 330]):
        # distance to a vane starting from the centre along direction ang
        along = xx * np.cos(ang) + yy * np.sin(ang)
        across = np.abs(-xx * np.sin(ang) + yy * np.cos(ang))
        p[(along > 0) & (across < spider)] = 0
    rng = rng or np.random.default_rng(0)
    rho, th = np.minimum(r, 1), np.arctan2(yy, xx)
    z = rng.normal(scale=0.08, size=4)  # waves rms-ish: defocus, astig x2, coma
    phase = (z[0] * (2 * rho ** 2 - 1) + z[1] * rho ** 2 * np.cos(2 * th)
             + z[2] * rho ** 2 * np.sin(2 * th) + z[3] * (3 * rho ** 3 - 2 * rho) * np.cos(th))
    return p, 2 * np.pi * phase


def hr_psf(amp, phase, shift=(0.0, 0.0)):
    """PSF on the HR grid, displaced by ``shift`` HR pixels (Fourier tilt)."""
    n = amp.shape[0]
    yy, xx = np.indices((n, n)) - n / 2
    tilt = 2 * np.pi * (shift[0] * yy + shift[1] * xx) / n
    field = amp * np.exp(1j * (phase + tilt))
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    return psf / psf.sum()


# Mesures de forme : module sprite.metrics (noms conservés pour les scripts).
moments_ellipticity = ellipticity
metrics = compare


def simulate(snr_db, rng, amp, phase):
    shifts = np.vstack([[0.0, 0.0], rng.uniform(-1, 1, size=(3, 2))])  # HR pixels
    truth = hr_psf(amp, phase)
    frames = np.array([hr_psf(amp, phase, s)[::2, ::2] for s in shifts])
    c = HR // 2
    signal_var = truth[c - 25:c + 25, c - 25:c + 25].var()
    sigma = np.sqrt(signal_var / 10 ** (snr_db / 10))
    frames = frames + rng.normal(scale=sigma, size=frames.shape)
    return truth, frames, shifts


METHODS = {
    "shift_and_add": None,
    "sprite_cpp": dict(cpp_compat=True),                       # run_sprite -F -N
    "sprite_fixed": dict(cpp_compat=False),                    # 3 quirks fixed, -F
    "sprite_fixed_noF": dict(cpp_compat=False, estimate_flux=False),
    # true shifts (run_sprite -S): isolates the registration error
    "sprite_fixed_noF_S": dict(cpp_compat=False, estimate_flux=False, true_shifts=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="starlet")
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--snr", type=float, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diam = HR / Q_HR
    results = {m: {str(s): [] for s in args.snr} for m in METHODS}
    t0 = time.time()
    for snr in args.snr:
        for r in range(args.n_real):
            rng = np.random.default_rng(1000 * int(snr) + r)
            amp, phase = pupil(HR, diam, rng=rng)
            truth, frames, shifts = simulate(snr, rng, amp, phase)
            for name, opts in METHODS.items():
                if opts is None:
                    res = sprite(frames, transform=args.transform, n_iter=1, n_reweight=0,
                                 kappa=4, estimate_flux=True, estimate_noise_per_frame=True)
                    est = lanczos_stack(frames, 2, res.shifts / 2, cpp_compat=False)
                else:
                    opts = dict(opts)
                    if opts.pop("true_shifts", False):
                        opts["shifts"] = shifts / 2  # LR pixels
                    opts.setdefault("estimate_flux", True)
                    est = sprite(frames, transform=args.transform, kappa=4,
                                 estimate_noise_per_frame=True, **opts).solution
                results[name][str(snr)].append(metrics(est, truth))
            print(f"SNR {snr:>4} dB  real {r + 1}/{args.n_real}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\ntransform={args.transform}  n_real={args.n_real}")
    header = f"{'method':<18}{'SNR':>5}" + "".join(f"{k:>12}" for k in ["rel_err", "e1_err", "e2_err", "fwhm_err"])
    print(header)
    summary = {}
    for name in METHODS:
        for snr in args.snr:
            rows = results[name][str(snr)]
            means = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
            summary.setdefault(name, {})[str(snr)] = means
            print(f"{name:<18}{snr:>5.0f}" + "".join(f"{means[k]:>12.4f}" for k in means))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "raw": results}, f, indent=1)


if __name__ == "__main__":
    main()
