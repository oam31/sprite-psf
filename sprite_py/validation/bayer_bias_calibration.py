"""Étude exploratoire : calibration du biais de l'estimateur de décalages Bayer (non intégrée).

Résultats : validation/results/bayer_bias_calibration.txt. Usage : PYTHONPATH=validation python validation/bayer_bias_calibration.py
"""
import time
import numpy as np
from scipy.ndimage import map_coordinates
import bayer_sim
from sprite import SpriteConfig, estimation
from sprite.bayer import _luminance, sublattices, sprite_bayer
from sprite.coadd import psf_sigma
from sprite.shifts import fourier_shift
from sprite.wavelets import Undecimated79
from sprite.metrics import compare

bayer_sim.DEFOCUS["R"] = 0.30
T = Undecimated79(6)

def lum_centroids(mosaic, psf_sig, sigma=None):
    lum = _luminance(mosaic, 'RGGB')
    sig = estimation.noise_sigma(lum) if sigma is None else sigma
    return estimation.estimate_centroids(lum, sig, 2.5 * psf_sig, 4.0), lum

def make_mosaic(models, shifts, up=2, ny=84):
    out = np.zeros((len(shifts), ny, ny)); pos = sublattices('RGGB')
    for k, s in enumerate(shifts):
        for b, offs in pos.items():
            img = fourier_shift(models[b], (up * s[0], up * s[1]))[::up, ::up]
            for oy, ox in offs: out[k, oy::2, ox::2] = img[oy::2, ox::2]
    return out

def bias_table(models, psf_sig, n=16, sigma_ref=None):
    """g(phi) - phi tabulated on a periodic grid of phases in [0,2)^2 (detector px)."""
    ph = np.arange(n) * 2.0 / n
    grid = np.array([[a, b] for a in ph for b in ph])
    frames = make_mosaic(models, np.vstack([[0, 0], grid]))
    # noise level for thresholds: same as real data (thresholds matter)
    sig = np.full(len(frames), sigma_ref)
    c, _ = lum_centroids(frames, psf_sig, sig)
    g = c[1:] - c[0]
    bias = (g - grid).reshape(n, n, 2)
    return bias

def correct(s_est, bias, n_iter=10):
    n = bias.shape[0]
    s = s_est.copy()
    for _ in range(n_iter):
        idx = np.mod(s, 2.0) / (2.0 / n)
        b = np.stack([map_coordinates(bias[..., a], idx.T, order=1, mode='grid-wrap') for a in (0, 1)], axis=1)
        s = s_est - b
    return s - s[0]

t0 = time.time()
truth, mosaic, _, pose = bayer_sim.simulate(500, 100, 2.5, 30)
cfg = SpriteConfig(estimate_noise_per_frame=True)
lum = _luminance(mosaic, 'RGGB')
sig_lum = estimation.noise_sigma(lum)
ps = psf_sigma(lum, 2, T)
c, _ = lum_centroids(mosaic, ps)
s_est = c - c[0]
rms = lambda s: float(np.sqrt(np.mean((s[1:] - pose[1:]) ** 2)))
print('luminance', round(rms(s_est), 4))
b_true = bias_table(truth, ps, sigma_ref=np.median(sig_lum))
print('corrigé, biais calculé sur PSF vraies', round(rms(correct(s_est, b_true)), 4), f'({time.time()-t0:.0f}s)')
shifts = s_est
for it in range(3):
    res = sprite_bayer(mosaic, cfg, pose_shifts=shifts)
    models = {b: r.solution for b, r in res.bands.items()}
    bias = bias_table(models, ps, sigma_ref=np.median(sig_lum))
    shifts = correct(s_est, bias)
    print(f'itération {it+1}: décalages {rms(shifts):.4f} px ; PSF (avec décalages précédents)',
          {b: round(compare(res.bands[b].solution, truth[b])['rel_err'], 3) for b in 'RGB'}, f'({time.time()-t0:.0f}s)', flush=True)
res = sprite_bayer(mosaic, cfg, pose_shifts=shifts)
print('final', {b: round(compare(res.bands[b].solution, truth[b])['rel_err'], 3) for b in 'RGB'})

from sprite.bayer import refine_pose_shifts
for it in range(3, 6):
    models = {b: r.solution for b, r in res.bands.items()}
    shifts = correct(s_est, bias_table(models, ps, sigma_ref=np.median(sig_lum)))
    res = sprite_bayer(mosaic, cfg, pose_shifts=shifts)
    print(f'itération {it+1}: décalages {rms(shifts):.4f} px ; PSF', {b: round(compare(res.bands[b].solution, truth[b])['rel_err'], 3) for b in 'RGB'}, f'({time.time()-t0:.0f}s)', flush=True)
models = {b: r.solution for b, r in res.bands.items()}
shifts2 = refine_pose_shifts(mosaic, models, shifts, 2)
res2 = sprite_bayer(mosaic, cfg, pose_shifts=shifts2)
print(f'+ réajustement : décalages {rms(shifts2):.4f} px ; PSF', {b: round(compare(res2.bands[b].solution, truth[b])['rel_err'], 3) for b in 'RGB'})
