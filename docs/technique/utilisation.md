# Utilisation et paramètres

## Utilisation

SPRITE s'utilise par la fonction Python `sprite()` ou par la commande `sprite-psf` ; la commande `sprite-psf-check` contrôle une reconstruction sur données réelles.

### Installation

```bash
cd sprite_py
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest          # 71 tests, environ 1 min 30
.venv/bin/python -m pytest --cov=sprite --cov-report=term-missing   # couverture par module
```

Dépendances : numpy, scipy, astropy, modopt, tifffile.

### API Python

```python
from sprite import sprite
from sprite.io import load_frames, load_bad_pixels

frames, dtypes = load_frames("images/*.tif")          # (n, ny, nx), float64
res = sprite(frames, estimate_noise_per_frame=True,
             subtract_background=True, saturation=65000,
             bad_pixels=load_bad_pixels("defauts.tif"))
res.solution        # PSF HR, (2 ny, 2 nx), alignée sur l'image de référence
res.shifts          # décalages (dy, dx) en pixels HR, image 0 = référence
res.flux, res.sigma # flux relatifs, bruit par image
res.residual        # résidu BR en unités des données (nul sur les pixels exclus)
res.valid, res.censored   # pixels valides, pixels saturés
```

La solution somme approximativement au flux de l'image de référence (`ref_frame`, 0 par défaut).

### Ligne de commande

```bash
# données réelles, réglages par défaut (starlet, kappa 4, centroid_wide)
sprite-psf -N --background --saturation 65000 --bad-pixels defauts.tif images/*.tif psf.tif sorties/

# même chose avec flux variables entre poses
sprite-psf -N -F --flux-method fit --flux-refine 3 --background images/*.tif psf.tif sorties/

# contrôle de la reconstruction (mêmes options, sans fichier de sortie)
sprite-psf-check -N --background --saturation 65000 --bad-pixels defauts.tif images/*.tif

# balayage de kappa par validation croisée
sprite-psf-check -N --background --scan kappa 2 3 4 5 6 8 images/*.tif

# images brutes en mosaïque de Bayer RGGB (non dématricées) : une PSF par bande
sprite-psf -N --bayer RGGB --background images/*.tif psf.tif sorties/

# contrôle d'une reconstruction Bayer (blocs et moitiés formés de poses entières)
sprite-psf-check -N --bayer RGGB --background images/*.tif

# configuration du papier reproduite à l'identique, erreurs du C++ comprises
sprite-psf -t 2 -s 4 -F -N --cpp-compat --cpp-orientation cube_test.fits solution.fits sorties/
```

Sorties de `sprite-psf` dans le dossier indiqué :

- la solution, au format de l'extension du fichier (`.fits` ou `.tif`, float32) ;
- `lr_<nom>` : les images BR prédites par la solution ;
- avec `-w fichier` : les résidus BR ;
- avec `-W` : les diagnostics, au format du C++ : `model` (première estimation), `pixel_map` (1 valide, 2 saturé, 0 exclu), `centroids.txt`, `shift.txt` (pixels HR, colonnes x puis y), `flux.txt` (flux relatifs, une ligne par image, réutilisable par `--fluxes`), `sig_est.txt`, `mse.txt`, `background.txt`, `param.txt`.

### Formats d'entrée

Chaque plan ou bande est une image BR. Les entrées acceptées sont :

- un cube FITS (n, ny, nx) ;
- un TIFF multipage ;
- un TIFF dont les échantillons par pixel ou les canaux sont les bandes ;
- plusieurs fichiers d'une image chacun, donnés en liste, en dossier ou par motif, et lus dans l'ordre.

Les échantillons TIFF peuvent être des entiers non signés 16 bits ou des flottants 32 bits ; les autres types entiers et flottants sont aussi acceptés. Tout est converti en float64. Les images doivent avoir la même taille et montrer la même étoile, à peu près centrée.

## Paramètres

Les paramètres sont regroupés par thème, de la même façon dans `SpriteConfig` (API) et dans l'aide de `sprite-psf`. La rétrocompatibilité, qui reproduit des erreurs du C++, est séparée du reste.

### Paramètres principaux

| Paramètre | Option CLI | Défaut | Conseillé | Rôle |
| --- | --- | --- | --- | --- |
| `transform` | `-t` | `starlet` (2) | `starlet` | dictionnaire d'ondelettes, voir « [Dictionnaires d'ondelettes](methode.md#dictionnaires-dondelettes--starlet-et-79) » |
| `kappa` | `-s` | 4 | 4, à confirmer par balayage | seuil de parcimonie en écarts types ; plus grand = solution plus lisse |
| `estimate_noise_per_frame` | `-N` | non | oui | bruit estimé image par image ; sinon, bruit maximal pour toutes |
| `upfact` | `-r` | 2 | 2 | facteur de sur-échantillonnage |

Les défauts sont les réglages mesurés comme les meilleurs ; `run_sprite` utilise la 7/9 et κ = 3. `-N` n'est pas activé par défaut, comme dans `run_sprite`, mais il est conseillé dès que le bruit peut varier d'une pose à l'autre. κ se règle sur les données par balayage (voir « [Converger vers le meilleur réglage](donnees-reelles.md#4-converger-vers-le-meilleur-réglage) »).

### Décalages

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `shift_method` | `--shift-method` | `centroid_wide` | estimateur : `centroid_wide`, `centroid` (méthode du papier, biaisée) ou `xcorr` |
| `wide_factor` | `--wide-factor` | 2,5 | fenêtre de `centroid_wide`, en largeurs de PSF |
| `shifts` | `-S` | estimés | décalages connus (fichier : une ligne `sx sy` par image, pixels BR) |
| `shift_refine` | `--shift-refine` | 0 | réajustements des décalages sur la solution (déconseillé, voir « [Estimation des décalages](etudes.md#estimation-des-décalages) ») |

### Flux

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `estimate_flux` | `-F` | non | estime les flux relatifs ; à n'utiliser que s'ils varient de plus de 3 % entre poses |
| `flux_method` | `--flux-method` | `aperture` | avec `-F` : `fit` (conseillé) ou `aperture` (méthode du papier) |
| `flux_refine` | `--flux-refine` | 0 | avec `-F` : réajustements sur la solution (3 conseillés) |
| `fluxes` | `--fluxes` | aucun | flux relatifs connus, une valeur par image ; prioritaires sur `-F` |

### Fond et pixels

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `subtract_background` | `--background` | non | soustrait la médiane de la bordure de chaque image |
| `saturation` | `--saturation` | aucune | niveau de saturation (données brutes) ; pixels traités comme bornes inférieures |
| `bad_pixels` | `--bad-pixels` | aucune | carte des pixels défectueux (non nul = défectueux) |
| `mask_dilate` | `--mask-dilate` | 0 | couronne exclue autour des zones saturées ou défectueuses, en pixels |

### Mosaïque de Bayer

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `pattern` (API `sprite_bayer`) | `--bayer` | aucun | motif de la mosaïque (`RGGB`, `BGGR`, `GRBG`, `GBRG`) ; active la reconstruction par bande |
| `upfact` | `-r` | 2 | facteur par rapport au pixel du détecteur, comme en monochrome |
| `pose_shifts` (API) | `-S` | estimés | décalages connus entre poses (fichier : une ligne `sx sy` par pose, pixels du détecteur) |
| `pose_refine` (API) | | 2 si ≥ 50 poses, sinon 0 | réajustements des décalages sur les PSF reconstruites |

### Exécution

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `n_jobs` (API `check`, `check_bayer`, `scan`, `sprite_bayer`) | `-j`, `--jobs` | cœurs moins un | nombre de processus pour les reconstructions indépendantes ; `1` pour tout exécuter dans le processus courant |

### Algorithme (avancé)

| Paramètre | Option CLI | Défaut | Rôle |
| --- | --- | --- | --- |
| `exact_adjoint` | `--prox-backprojection` | auto | prox d'analyse : adjoint exact ou reconstruction ; auto = adjoint pour la starlet, reconstruction pour la 7/9 (voir « [Prox d'analyse : adjoint exact](etudes.md#prox-danalyse--adjoint-exact) ») |
| `n_scales` | `-n` | ⌊log₂ N<sub>HR</sub>⌋ − 1 | nombre d'échelles des ondelettes |
| `n_iter` | `-i` | 50 | itérations maximales par passe |
| `n_reweight` | `-R` | 1 | passes de repondération après la première |
| `cv_win`, `cv_tol` | `-C`, `-a` | 25, 0,005 | critère d'arrêt à fenêtre glissante |
| `w1`, `relax` | (API) | 0,5, 1,4 | poids et relaxation du forward-backward généralisé |
| `lanczos_radius` | (API) | 10 | rayon des noyaux de recalage de l'opérateur |
| `first_guess_nsig` | (API) | 5 | seuil du débruitage de la première estimation |
| `shift_nsig` | (API) | 4 | seuil de bruit pour l'estimation des décalages |
| `ref_frame` | (API) | 0 | image de référence (décalage nul, flux 1) |
| `psf_sigma`, `spectral_radius` | (API) | estimés | imposent la largeur de PSF ou le rayon spectral |

### Rétrocompatibilité avec le C++ publié

Ces options reproduisent des **erreurs** du code d'origine, pour retrouver ses résultats à l'identique (comparaison avec le papier, non-régression). Elles ne doivent pas servir à des mesures.

| Paramètre | Option CLI | Erreur reproduite |
| --- | --- | --- |
| `cpp_compat` | `--cpp-compat` | les quatre erreurs de « [Erreurs du C++ corrigées](implementation.md#erreurs-du-c-corrigées) », plus l'estimateur de décalages `centroid` du papier |
| `compat_stack` | `--cpp-compat-stack` | recalage de la première estimation au décalage opposé, sur les axes permutés |
| `compat_flux` | `--cpp-compat-flux` | ouverture de flux décalée d'un pixel sur chaque axe ; empilement des images sans correction de flux |
| `compat_stopping` | `--cpp-compat-stopping` | moyenne glissante du critère d'arrêt mal mise à jour |
| (par `cpp_compat`) | `--cpp-compat` | reconstruction à la place de l'adjoint dans le prox d'analyse, y compris pour la starlet |
| (sorties) | `--cpp-orientation` | images 2D écrites transposées |

`--cpp-compat` ne change pas le dictionnaire ni κ : pour retrouver exactement une commande `run_sprite`, reprendre ses options (`-t 24 -s 3` pour ses valeurs par défaut, `-t 2 -s 4` ou `-t 24 -s 4` pour les commandes du papier).

Les méthodes du papier qui ne sont pas des erreurs, mais simplement moins bonnes, restent des options ordinaires : `--shift-method centroid`, `--flux-method aperture`, et la reconstruction dans le prox pour la 7/9, conservée par défaut parce qu'elle y donne de meilleurs résultats.

## Traitement des données

### Soustraction du fond

Avec `--background` (API : `subtract_background=True`), chaque image perd une constante : la médiane des pixels valides de sa bordure. La soustraction précède toute autre étape, y compris l'estimation du bruit.

- **Bordure** : bande de `w` pixels le long des quatre côtés, avec `w` = 10 % du plus petit côté de l'image, et au moins 2 pixels. Pour une image de 84 × 84, `w` = 8, soit 2 432 pixels sur 7 056.
- **Estimateur** : la médiane, qui ignore les quelques pixels chauds ou rayons cosmiques présents dans la bordure ; les pixels saturés ou défectueux n'y entrent pas.
- **Indépendance** : une valeur par image, car l'offset peut varier d'une pose à l'autre.
- **Trace** : les valeurs soustraites sont renvoyées dans `res.background` et écrites dans `background.txt` avec `-W`.

La méthode suppose un fond uniforme sur l'image, une bordure sans flux de l'étoile et un bruit de fond symétrique. Pour des vignettes serrées ou un fond non uniforme, un fond mesuré sur des poses sombres ou une région vide du détecteur reste préférable ; il se soustrait avant l'appel à `sprite()`.

### Pixels saturés ou défectueux

Un pixel défectueux est exclu de l'attache aux données ; un pixel saturé y reste, comme borne inférieure. Un pixel défectueux ne porte aucune information ; un pixel saturé dit que la vraie valeur est au moins égale au niveau de saturation. Avec r = y − M(x⁽⁰⁾ + Δ) le résidu blanchi :

```math
J_1 = \frac{1}{2} \sum_{\text{valides}} r^2 + \frac{1}{2} \sum_{\text{saturés}} \max(r, 0)^2
```

- **Pixels défectueux** : carte fournie (`--bad-pixels`, FITS ou TIFF, commune à toutes les images ou une par image) et valeurs non finies. Poids nul : la solution ne dépend pas de leur valeur.
- **Pixels saturés** : valeur brute ≥ `--saturation`. Le terme censuré ne pénalise le modèle que s'il passe sous la valeur mesurée. Il reste convexe et dérivable, de gradient lipschitzien de même constante : l'algorithme et son pas sont inchangés.
- **Couronne** (`--mask-dilate N`) : les pixels à moins de N pixels d'une zone saturée ou défectueuse sont exclus (débordement de charge, diaphonie).
- **Estimations préalables** (bruit, largeur de PSF, décalages, flux, première estimation) : elles demandent des images complètes. Les pixels exclus y sont remplacés par une interpolation harmonique (moyenne itérée des voisins, exacte sur une variation linéaire) ; les pixels saturés gardent leur valeur écrêtée. L'interpolation n'entre jamais dans l'attache aux données.

Résultats sur données simulées (30 dB, dictionnaire 7/9, 3 réalisations, `validation/results/masking_and_snr.txt`) :

| Cas | Erreur relative | Erreur de FWHM |
| --- | --- | --- |
| Données propres | 0,099 | 1,2 % |
| 0,3 % de pixels chauds (+50 σ), sans carte | 0,504 | 1,1 % |
| 0,3 % de pixels chauds, avec carte | 0,099 | 1,2 % |
| Saturation à 80 % du pic (5 pixels), non signalée | 0,257 | 15,1 % |
| Saturation à 80 % du pic, pixels exclus (poids nul) | 0,586 | 68,1 % |
| Saturation à 80 % du pic, terme censuré (retenu) | 0,236 | 12,4 % |

Les pixels défectueux sont entièrement neutralisés. Pour la saturation, exclure les pixels est pire que ne rien faire, car plus rien ne contraint la hauteur du pic ; le terme censuré améliore le résultat sans le ramener au niveau des données propres. Un pic saturé dans toutes les images reste une perte d'information : mieux vaut des poses plus courtes.
