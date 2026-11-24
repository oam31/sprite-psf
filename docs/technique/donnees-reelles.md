# Mise en œuvre sur données réelles


Sans PSF de référence, une reconstruction se valide par sa cohérence avec les données : ajustement au niveau du bruit, capacité à prédire des poses qu'elle n'a pas vues, stabilité entre sous-ensembles de poses. La commande `sprite-psf-check` calcule ces contrôles ; les seuils ci-dessous viennent de leur calibration sur données simulées. Pour les mosaïques de Bayer (`--bayer`), voir aussi « [Contrôle d'une reconstruction Bayer](bayer.md#contrôle-dune-reconstruction-bayer) ».

## 1. Préparer les données

- **Vignettes** : même taille pour toutes les poses, étoile à peu près centrée, au moins 10 à 20 pixels de marge autour du cœur pour le fond et les ailes.
- **Fond** : `--background` si les images ont un offset (données entières de détecteur) ; sinon soustraire un fond mesuré (pose sombre, biais) avant.
- **Pixels** : niveau de saturation du détecteur (`--saturation`, en unités brutes, avant soustraction du fond) et carte des pixels défectueux (`--bad-pixels`).
- **Flux** : si les temps de pose ou un moniteur de source sont connus, préparer un fichier de flux relatifs pour `--fluxes`.
- **Diversité sous-pixel** : les poses doivent couvrir des phases variées. Après une première reconstruction, vérifier dans `shift.txt` (pixels HR) que les parties fractionnaires des décalages BR (décalages HR / 2) se répartissent sur [0, 1) dans les deux directions ; des poses toutes à la même phase ne permettent pas la super-résolution.

## 2. Reconstruire avec les réglages conseillés

```bash
sprite-psf -N --background --saturation NIVEAU --bad-pixels carte.tif \
    -W -w residus.fits images/*.tif psf.tif sorties/
```

Puis examiner :

- `residus.fits` : les résidus doivent ressembler à du bruit, sans motif à la position de l'étoile ; une couronne ou une asymétrie signale un fond, des décalages ou des flux mal estimés ;
- `pixel_map` : vérifier que les pixels saturés et exclus sont bien ceux attendus ;
- `flux.txt`, `shift.txt` : valeurs plausibles, pas d'image aberrante.

## 3. Contrôler

```bash
sprite-psf-check -N --background --saturation NIVEAU --bad-pixels carte.tif images/*.tif
```

Il faut au moins 8 poses. Le contrôle refait la reconstruction, puis calcule (les reconstructions partielles sont réparties sur `-j` processus ; 13 s au lieu de 24 s pour 50 poses de 84 × 84 pixels sur 10 cœurs) :

1. le **χ²/N d'ajustement** par image, sur les pixels valides ;
2. le **χ²/N de prédiction** : les poses sont réparties en 5 blocs, et chaque bloc est prédit par une reconstruction faite sans lui ;
3. l'**écart entre moitiés** : deux reconstructions sur les poses paires et impaires ;
4. la **dispersion apparente des flux** : écart type des flux ajustés sur la solution.

Calibration sur données simulées (starlet, `-N`, κ = 4 sauf mention, 2 réalisations, `validation/results/check_calibration.txt`) :

| Cas | χ²/N ajustement | χ²/N prédiction | Rapport | Écart entre moitiés | Dispersion des flux | Erreur vraie |
| --- | --- | --- | --- | --- | --- | --- |
| Correct, 16 images, 30 dB | 0,957 | 0,968 | 1,01 | 0,022 | 0,2 % | 0,056 |
| Correct, 16 images, 20 dB | 0,965 | 0,972 | 1,01 | 0,061 | 0,5 % | 0,110 |
| Correct, 64 images, 30 dB | 0,948 | 0,954 | 1,01 | 0,009 | 0,2 % | 0,032 |
| Décalages du papier (`centroid`) | 1,907 | 2,223 | 1,17 | 0,204 | 6,1 % | 0,224 |
| Flux ±20 % non estimés | 1,481 | 2,038 | 1,36 | 0,144 | 9,5 % | 0,070 |
| Flux ±20 % estimés (fit + 3) | 0,960 | 0,971 | 1,01 | 0,022 | 10,5 % (vrais) | 0,057 |
| Régularisation faible (κ = 1) | 0,864 | 1,016 | 1,18 | 0,046 | 0,2 % | 0,244 |
| Régularisation forte (κ = 8) | 0,972 | 0,982 | 1,01 | 0,031 | 0,2 % | 0,083 |

Lecture :

- **χ²/N d'ajustement proche de 1** (0,9 à 1,1) : le modèle explique les données. Au-dessus de 1,3 : quelque chose manque (fond, pixels défectueux ou saturation non signalés, flux variables, décalages faux). Sous 0,8 : bruit surestimé (bruit corrélé, images rééchantillonnées ou prétraitées) ou sur-ajustement.
- **Rapport prédiction / ajustement proche de 1** (≤ 1,1) : la solution généralise. Au-dessus : régularisation trop faible, ou paramètres d'image faux.
- **Dispersion des flux** : 0,2 à 0,5 % pour des flux égaux ; au-delà de 3 %, les flux varient et doivent être estimés ou fournis. Quand les flux sont déjà estimés, elle mesure leur variation réelle.
- **Écart entre moitiés** : erreur due au bruit seul. Il sous-estime l'erreur vraie (d'un facteur 2 à 3 ici), car il ne voit pas le biais de la régularisation, commun aux deux moitiés.
- **Images atypiques** : une pose dont le χ²/N dépasse nettement les autres (signalée dans le rapport) est à examiner, voire à retirer (étoile voisine, rayon cosmique sur le cœur, pose bougée).

## 4. Converger vers le meilleur réglage

Les réglages se fixent dans un ordre précis : d'abord ce qui rend le modèle juste (fond, pixels, flux), ensuite seulement la régularisation. Un κ réglé sur un modèle faux compense une erreur au lieu de la corriger.

| Ordre | Paramètre | Indicateur à surveiller | Symptôme d'un mauvais réglage | Action |
| --- | --- | --- | --- | --- |
| 1 | `--background` | résidus en bordure, `background.txt` | χ²/N > 1,3 avec des résidus de même signe partout | activer, ou soustraire un fond mesuré |
| 2 | `--saturation`, `--bad-pixels`, `--mask-dilate` | `pixel_map`, résidus, images atypiques | pic sous-estimé, χ²/N élevé sur quelques images, pixels isolés dans les résidus | signaler les pixels ; élargir la couronne si les résidus entourent les zones saturées |
| 3 | `-F`, `--fluxes` | dispersion apparente des flux, χ²/N | dispersion > 3 %, χ²/N > 1,3, rapport prédiction / ajustement > 1,1 | fournir les flux ou `-F --flux-method fit --flux-refine 3` |
| 4 | `--shift-method`, `--wide-factor` | χ²/N, `shift.txt`, rapport | χ²/N élevé sur toutes les images, phases sous-pixel groupées | garder `centroid_wide` ; essayer `--wide-factor 2` à faible SNR, 3 à fort SNR |
| 5 | `-s` (κ) | `--scan kappa`, rapport, écart entre moitiés | rapport > 1,1 et χ²/N d'ajustement < 0,9 : κ trop faible | balayer κ et retenir la valeur proposée |
| 6 | `-t` | `--scan transform` | | comparer starlet et 7/9 si la PSF a des structures alignées sur la grille |
| 7 | `-N` | χ²/N par image | χ²/N très différent d'une image à l'autre | activer |

À chaque étape, relancer `sprite-psf-check` : le χ²/N d'ajustement doit se rapprocher de 1 et le rapport prédiction / ajustement rester sous 1,1.

**Balayage de κ.** `sprite-psf-check --scan kappa 2 3 4 5 6 8` reconstruit pour chaque valeur et mesure le χ²/N de prédiction, avec son erreur type estimée sur la dispersion entre images. Le minimum de ce critère est un mauvais guide à lui seul : la prédiction est peu sensible à une régularisation trop faible, et son minimum tombe souvent un cran sous la meilleure valeur. La commande retient donc la plus grande valeur dont le χ²/N de prédiction reste à moins d'une erreur type du minimum (règle « une erreur type »). Vérification sur données simulées (16 images, starlet, `-N`, `validation/results/scan_calibration.txt`) :

| Cas | κ du minimum | κ retenu (erreur vraie) | Meilleur κ vrai (erreur vraie) | Erreur avec κ = 4 |
| --- | --- | --- | --- | --- |
| 20 dB, réalisation 1 | 3 | 6 (0,125) | 4 (0,124) | 0,124 |
| 20 dB, réalisation 2 | 3 | 6 (0,111) | 4 (0,104) | 0,104 |
| 30 dB, réalisation 1 | 2 (0,074) | 4 (0,046) | 3 (0,043) | 0,046 |
| 30 dB, réalisation 2 | 3 | 4 (0,061) | 3 (0,060) | 0,061 |

La règle reste à moins de 7 % de l'erreur optimale, alors que le simple minimum peut la dépasser de 72 % (κ = 2 dans le troisième cas). L'erreur varie peu entre κ = 3 et κ = 6 : le défaut κ = 4 est un bon point de départ, et le balayage sert surtout à écarter les valeurs franchement mauvaises. Les écarts de χ²/N de prédiction entre valeurs sont de l'ordre de 0,1 à 1 % : avec moins de 10 poses, l'erreur type devient du même ordre et le balayage perd son pouvoir de décision.

**Dictionnaire.** `--scan transform 2 24` applique le même critère (valeur du minimum, sans règle d'erreur type).

## 5. Ce que les données réelles ne permettent pas de valider

Sur données simulées, la PSF vraie est connue : on mesure directement l'erreur de la reconstruction. Sur données réelles, elle ne l'est pas, et aucun des contrôles ci-dessus ne mesure cette erreur. Ils vérifient la **cohérence** de la solution avec les données, pas son **exactitude** :

- **Un χ²/N proche de 1 est nécessaire, pas suffisant.** Il dit que le modèle reproduit les images BR au niveau du bruit ; plusieurs PSF HR différentes peuvent le faire, surtout dans les hautes fréquences que la super-résolution cherche justement à retrouver.
- **Les biais communs à toutes les poses sont invisibles.** La prédiction croisée et la comparaison entre moitiés ne détectent que ce qui varie d'une pose à l'autre. Un écart du modèle partagé par toutes les poses passe inaperçu : réponse du pixel et sensibilité intrapixel non modélisées, non-linéarité du détecteur, bruit corrélé, couleur de la source, biais propre à la régularisation (lissage du cœur, atténuation des ailes).
- **L'écart entre moitiés ne mesure que le bruit.** Il sous-estime l'erreur vraie d'un facteur 2 à 3 sur données simulées.
- **Les seuils de ce document viennent de simulations sans réponse du pixel ni bruit de photons.** Ils donnent des ordres de grandeur ; sur données réelles, ce sont les tendances (amélioration ou dégradation d'un indicateur quand on change un réglage) qui comptent, plus que les valeurs absolues.
- **Une PSF qui varie entre poses** (dérive de focalisation, agitation, turbulence) viole l'hypothèse de SPRITE ; elle se traduit par des images atypiques ou un χ²/N élevé, mais la solution obtenue est une moyenne, pas la PSF d'une pose.

Pour s'approcher d'une validation par simulation, deux moyens restent possibles :

- **Recoupement indépendant** : comparer la FWHM et les ellipticités de la solution, que le rapport de contrôle indique, à une mesure indépendante (PSF mesurée avec un meilleur échantillonnage, modèle optique de l'instrument, mesure interférométrique).
- **Injection** : ajouter à des images réelles sans étoile (ou à des zones vides des vignettes) une PSF simulée connue, avec les mêmes décalages et flux que les poses réelles, puis la reconstruire. On retrouve alors une vérité terrain avec le bruit, le fond et les défauts réels du détecteur. Cet outil n'est pas encore écrit (voir « [Pistes](pistes.md) »).
