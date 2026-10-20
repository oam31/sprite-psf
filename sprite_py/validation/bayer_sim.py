"""Validation de la reconstruction par bande sur mosaïque de Bayer RGGB simulée.

PSF chromatiques : même pupille et mêmes aberrations (différence de marche
fixe, donc phase en lambda_G / lambda), diffraction à l'échelle de chaque
longueur d'onde, plus un décalage latéral et une défocalisation propres à
chaque bande. Le détecteur échantillonne G à 0,688 × Nyquist (comme le cas
monochrome de référence) ; la grille HR a un pas d'un demi-pixel du détecteur.

Les poses sont décalées aléatoirement de ±a pixel du détecteur. On compare :
chaque bande reconstruite depuis la mosaïque, et la bande G reconstruite depuis
un détecteur monochrome (référence sans perte d'échantillonnage).

Usage : python validation/bayer_sim.py [--n-poses 16 50] [--amp 0.5 1.0] [--snr 30]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sprite import SpriteConfig, sprite
from sprite.bayer import _centroid, sprite_bayer, sublattices
from sprite.metrics import compare
from synthetic import HR, Q_HR, hr_psf, pupil

LAMBDA = {"R": 0.62, "G": 0.53, "B": 0.46}
LATERAL = {"R": (0.3, 0.4), "G": (0.0, 0.0), "B": (-0.3, -0.2)}  # pixels HR
DEFOCUS = {"R": 0.10, "G": 0.0, "B": -0.10}  # vagues à lambda_G


def band_psfs(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Amplitude et phase pupillaires de chaque bande (mêmes aberrations)."""
    out = {}
    for band, lam in LAMBDA.items():
        q = Q_HR * lam / LAMBDA["G"]
        amp, phase = pupil(HR, HR / q, rng=np.random.default_rng(seed))
        yy, xx = (np.indices((HR, HR)) - HR / 2) / (HR / q / 2)
        rho2 = np.minimum(yy ** 2 + xx ** 2, 1.0)
        defocus = 2 * np.pi * DEFOCUS[band] * (2 * rho2 - 1)
        out[band] = (amp, (phase + defocus) * LAMBDA["G"] / lam)
    return out


def simulate(seed: int, n_poses: int, amp: float, snr_db: float):
    rng = np.random.default_rng(seed + 1)
    psfs = band_psfs(seed)
    truth = {b: hr_psf(a, p, LATERAL[b]) for b, (a, p) in psfs.items()}
    pose = np.vstack([[0.0, 0.0], rng.uniform(-amp, amp, size=(n_poses - 1, 2))])  # pixels détecteur
    ny = HR // 2
    mosaic = np.zeros((n_poses, ny, ny))
    mono_g = np.zeros((n_poses, ny, ny))
    pos = sublattices("RGGB")
    for k, s in enumerate(pose):
        full = {b: hr_psf(a, p, np.add(LATERAL[b], 2 * s))[::2, ::2] for b, (a, p) in psfs.items()}
        mono_g[k] = full["G"]
        for b, offs in pos.items():
            for oy, ox in offs:
                mosaic[k, oy::2, ox::2] = full[b][oy::2, ox::2]
    c = HR // 2
    sigma = np.sqrt(truth["G"][c - 25:c + 25, c - 25:c + 25].var() / 10 ** (snr_db / 10))
    mosaic += rng.normal(scale=sigma, size=mosaic.shape)
    mono_g += rng.normal(scale=sigma, size=mono_g.shape)
    return truth, mosaic, mono_g, pose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-poses", type=int, nargs="+", default=[16, 50])
    ap.add_argument("--amp", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--snr", type=float, default=30)
    ap.add_argument("--n-real", type=int, default=2)
    ap.add_argument("--upfact", type=int, nargs="+", default=[2, 1])
    args = ap.parse_args()

    print(f"SNR {args.snr} dB, starlet, kappa 4, -N ; {args.n_real} réalisations ; erreur relative / erreur de FWHM.")
    print("r = facteur par rapport au pixel du détecteur (2 : pas HR d'un demi-pixel ; 1 : pas du détecteur).")
    print(f"{'r':>2} {'poses':>5} {'±a':>4}  {'G mono':>15}  {'R Bayer':>15}  {'G Bayer':>15}  {'B Bayer':>15}"
          f"  {'R vrais déc.':>15}  {'G vrais déc.':>15}  {'B vrais déc.':>15}  {'err. déc.':>9}  {'err. R-G':>8}  {'err. B-G':>8}")
    t0 = time.time()
    for up in args.upfact:
        cfg = SpriteConfig(estimate_noise_per_frame=True, upfact=up)
        for n in args.n_poses:
            for a in args.amp:
                acc = {k: [] for k in ("mono", "R", "G", "B", "Rt", "Gt", "Bt", "pose", "RG", "BG")}
                for r in range(args.n_real):
                    truth, mosaic, mono_g, pose = simulate(100 * r + 7, n, a, args.snr)
                    if up == 1:  # référence au pas du détecteur : PSF HR échantillonnée aux pixels
                        truth = {b: t[::2, ::2] for b, t in truth.items()}
                    acc["mono"].append(compare(sprite(mono_g, cfg).solution, truth["G"]))
                    res = sprite_bayer(mosaic, cfg)
                    ref = sprite_bayer(mosaic, cfg, pose_shifts=pose)
                    for b in "RGB":
                        acc[b].append(compare(res.bands[b].solution, truth[b]))
                        acc[b + "t"].append(compare(ref.bands[b].solution, truth[b]))
                    acc["pose"].append(float(np.sqrt(np.mean((res.pose_shifts[1:] - pose[1:]) ** 2))))
                    tc = {b: _centroid(truth[b]) for b in "RGB"}
                    acc["RG"].append(float(np.linalg.norm(res.chromatic["R-G"] - (tc["R"] - tc["G"]))))
                    acc["BG"].append(float(np.linalg.norm(res.chromatic["B-G"] - (tc["B"] - tc["G"]))))
                cell = lambda ms: (f"{np.mean([m['rel_err'] for m in ms]):.3f} / "
                                   f"{100 * np.mean([m['fwhm_err'] for m in ms]):4.1f} %")
                print(f"{up:>2} {n:>5} {a:>4.1f}  " + "  ".join(f"{cell(acc[k]):>15}" for k in
                      ("mono", "R", "G", "B", "Rt", "Gt", "Bt")) +
                      f"  {np.mean(acc['pose']):>6.3f} px  {np.mean(acc['RG']):>8.3f}  {np.mean(acc['BG']):>8.3f}"
                      f"   ({time.time() - t0:.0f} s)", flush=True)

if __name__ == "__main__":
    main()
