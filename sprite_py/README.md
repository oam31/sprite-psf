# sprite-psf : portage Python de SPRITE

Super-résolution de PSF par régularisation parcimonieuse, d'après
F. M. Ngolè Mboula, J.-L. Starck, S. Ronayette, K. Okumura, J. Amiaux,
*Super-resolution method using sparse regularization for point-spread function
recovery*, A&A 575, A86 (2015), arXiv:1410.7679.

Portage du code C++ SPRITE v1.0 (CEA, 2014, licence CeCILL) en Python, avec le
forward-backward généralisé de [ModOpt](https://github.com/CEA-COSMIC/ModOpt).

Portage indépendant et non officiel, ni développé ni approuvé par le CEA ou
l'équipe CosmoStat.

## Installation

```bash
pip install "git+https://github.com/oam31/sprite-psf.git#subdirectory=sprite_py"
```

ou, depuis une copie locale de ce dossier :

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
```

Dépendances : numpy, scipy, astropy, modopt, tifffile. Le code est typé et documenté (docstrings numpy en français) ; la documentation complète est dans `../docs/technique/reference.md`.

## Utilisation

```python
from sprite import sprite
from sprite.io import load_frames

frames, dtypes = load_frames("images/*.tif")     # (n_frames, ny, nx)
res = sprite(frames, estimate_noise_per_frame=True,          # starlet, kappa 4 par défaut
             subtract_background=True, saturation=65000)
res.solution      # PSF super-résolue (ny*2, nx*2)
```

En ligne de commande (réglages conseillés) puis contrôle sur données réelles :

```bash
sprite-psf -N --background --saturation 65000 --bad-pixels defauts.tif images/*.tif psf.tif sorties/
sprite-psf-check -N --background --saturation 65000 --bad-pixels defauts.tif images/*.tif
sprite-psf-check -N --background --scan kappa 2 3 4 5 6 8 images/*.tif
sprite-psf -N --bayer RGGB --background images/*.tif psf.tif sorties/   # mosaïque brute
sprite-psf-check -N --bayer RGGB --background images/*.tif
```

Les reconstructions indépendantes (bandes, blocs et moitiés du contrôle, valeurs
balayées) sont réparties sur plusieurs processus : option `-j` (défaut : cœurs
moins un).

`sprite-psf --help` présente les options par thème ; le groupe
« rétrocompatibilité avec le C++ publié » réunit les options qui reproduisent
les erreurs du code d'origine (`--cpp-compat`…), à ne pas utiliser pour des
mesures.

La documentation complète (méthode, dictionnaires, paramètres, réglage et
validation sur données réelles, résultats) est dans
[`../docs/technique/reference.md`](../docs/technique/reference.md).

## Organisation

| Module | Rôle | Équivalent C++ |
|---|---|---|
| `wavelets.py` | starlet (`-t 2`), 7/9 non décimée (`-t 24`), MAD iSAP | `MR_Trans`, `SB_Filter`, `get_sigma_mad` |
| `operators.py` | noyaux de Lanczos, M et Mᵀ (FFT), rayon spectral | `sprite_op_mat`, `sprite_transp_op_mat`, `spectral_rad_est` |
| `io.py` | lecture FITS/TIFF, écriture, fond de bord | `io_3d_read_data` |
| `estimation.py` | bruit, centroïdes, flux | `noise_est`, `compute_centroid`, `flux_est` |
| `masking.py` | pixels saturés (censurés) et défectueux (exclus) | (aucun) |
| `metrics.py` | FWHM, ellipticités | (aucun) |
| `diagnostics.py` | contrôles sur données réelles (`sprite-psf-check`), monochromes et Bayer | (aucun) |
| `bayer.py` | mosaïque de Bayer : sous-réseaux, décalages des poses, une reconstruction par bande | (aucun) |
| `parallel.py` | répartition des reconstructions indépendantes sur des processus | (aucun) |
| `shifts.py` | estimateurs de décalage, réajustement sur un modèle | `shift_est` |
| `coadd.py` | taille de PSF (co-addition iSAP et ajustement gaussien) | `lr_psf_size`, `coadd_frames` |
| `first_guess.py` | shift-and-add de Lanczos et support multirésolution | `mod_gen_2`, `mr_support_filt` |
| `prox.py` | prox d'analyse pondéré (FISTA dual), contrainte δ ≥ −x⁽⁰⁾ | `wl_filter_analysis`, `ineq_cons` |
| `solver.py` | algorithme 2 du papier sur `modopt.GenForwardBackward` | `Sprite::min2`, `do_it` |
| `cli.py` | commandes `sprite-psf` et `sprite-psf-check` | `run_sprite` |

## Contrôle et résultats

Le portage reproduit le C++ d'origine à 10⁻⁶ près en mode `--cpp-compat`
(référence recompilée dans `../reference`), et 71 tests (97 % du code) le vérifient étage par
étage. Les études sur données simulées (décalages, adjoint, flux, pixels,
nombre d'images) et leurs scripts sont dans `validation/` ; leurs résultats et
leur interprétation sont dans le document technique.
