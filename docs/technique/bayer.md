# Mosaïque de Bayer


La reconstruction par bande d'images Bayer brutes atteint la qualité monochrome si les poses couvrent au moins une période du motif (décalages de ±1 pixel ou plus) et si leurs décalages sont exacts. Estimés par SPRITE sur la mosaïque, les décalages gardent une erreur d'environ 0,06 à 0,08 pixel ; avec 100 poses décalées de plusieurs pixels, les PSF R et G restent alors à moins de 0,1 d'erreur relative, et B, la plus sensible, vers 0,1 à 0,2.

## Principe

Une mosaïque RGGB échantillonne chaque bande sur un sous-réseau du détecteur : sur chaque bloc de 2 × 2 pixels, R en (0, 0), G en (0, 1) et (1, 0), B en (1, 1). Chaque sous-réseau a un pas de 2 pixels : R et B sont deux fois plus sous-échantillonnés qu'avec une matrice monochrome, et G un peu moins grâce à ses deux sous-réseaux. Les PSF diffèrent d'une bande à l'autre : la diffraction les élargit avec la longueur d'onde, et les aberrations chromatiques ajoutent un décalage latéral et une défocalisation propres à chaque bande.

Le modèle de SPRITE s'étend sans changement de principe : y<sub>k</sub><sup>c</sup> = f<sub>k</sub><sup>c</sup> D<sub>c</sub> H<sub>k</sub> x<sub>c</sub>, où seule la décimation D<sub>c</sub> dépend de la bande. Les images ne doivent pas être dématricées : le dématriçage interpole les pixels manquants, ce qui fausse le modèle d'observation.

## Mise en œuvre (`--bayer RGGB`)

1. **Découpage** : chaque pose est découpée en quatre sous-images (R, G1, G2, B) de pas 2 pixels, chacune avec un décalage connu dans le bloc 2 × 2. Le sous-réseau de position o voit une pose décalée de s pixels du détecteur décalée de (s − o)/2 pixels du sous-réseau.
2. **Décalages entre poses** : le déplacement de l'étoile ne dépend pas de la couleur ; il est estimé une seule fois, par `centroid_wide` sur une pseudo-luminance (chaque bande de la mosaïque divisée par son flux moyen), puis appliqué à toutes les bandes. À partir de 50 poses, il est ensuite réajusté deux fois sur les PSF reconstruites (voir « Estimation des décalages sur la mosaïque »).
3. **Reconstruction** : chaque bande est reconstruite par SPRITE avec un facteur 2r sur son sous-réseau, donc sur la même grille HR que la matrice monochrome (pas de 1/r pixel du détecteur). Les deux sous-réseaux verts entrent comme deux images par pose. Fond, bruit, flux, saturation et pixels défectueux sont traités par bande.
4. **Diagnostic chromatique** : les décalages étant communs, un décalage chromatique latéral apparaît comme une translation entre les PSF reconstruites. Le rapport `chromatique.txt` donne, par bande, le centroïde, la FWHM et les ellipticités, ainsi que les décalages R − G et B − G.

Sorties : `psf_R`, `psf_G`, `psf_B` (au format de l'extension), `chromatique.txt`, et avec `-W` les flux et bruits par bande et `pose_shift.txt`.

Les trois bandes sont reconstruites en parallèle (`-j`). Le contrôle `sprite-psf-check --bayer` est décrit dans « Contrôle d'une reconstruction Bayer ».

## Résultats sur données simulées

Simulation (`validation/bayer_sim.py`, `validation/results/bayer_sim.txt`) : même pupille et mêmes aberrations pour les trois bandes (R 0,62 µm, G 0,53 µm, B 0,46 µm), diffraction à l'échelle de chaque longueur d'onde, décalages latéraux R (+0,3 ; +0,4) et B (−0,3 ; −0,2) pixel HR, défocalisation de ±0,1 onde ; G échantillonné à 0,688 × Nyquist comme en monochrome ; 30 dB ; 2 réalisations. Erreur relative / erreur de FWHM, pas HR d'un demi-pixel du détecteur (r = 2) :

| Poses, décalages | G monochrome | R Bayer | G Bayer | B Bayer | Erreur sur les décalages |
| --- | --- | --- | --- | --- | --- |
| 50, ±1 px, décalages vrais | 0,036 / 0,1 % | 0,054 / 0,2 % | 0,044 / 0,2 % | 0,058 / 0,8 % | 0 |
| 50, ±1 px, décalages estimés | | 0,136 / 5,0 % | 0,100 / 0,2 % | 0,176 / 9,2 % | 0,078 px |
| 50, ±0,5 px, décalages vrais | 0,039 / 0,1 % | 0,226 / 3,5 % | 0,233 / 18,2 % | 0,746 / 22,0 % | 0 |
| 50, ±0,5 px, décalages estimés | | 0,216 / 0,9 % | 0,205 / 11,1 % | 0,673 / 21,8 % | 0,082 px |
| 16, ±1 px, décalages vrais | 0,062 / 0,1 % | 0,165 / 5,9 % | 0,068 / 0,1 % | 0,254 / 26,3 % | 0 |
| 16, ±1 px, décalages estimés | | 0,200 / 7,9 % | 0,121 / 1,1 % | 0,343 / 39,0 % | 0,071 px |

Ce qu'il faut en retenir :

- **Amplitude des décalages entre poses.** La super-résolution exploite les positions différentes de l'étoile par rapport à la grille des pixels. En monochrome, la grille a une période de 1 pixel : des décalages de l'ordre du pixel suffisent. Sur les sous-réseaux R et B, la période est de 2 pixels : des décalages de ±0,5 pixel n'en couvrent que la moitié dans chaque direction, et certaines composantes de la PSF ne sont jamais observées. Il faut des décalages d'au moins ±1 pixel, idéalement répartis uniformément sur une période ; s'ils sont pilotables, des pas volontaires de 1 à 2 pixels entre poses sont la mesure la plus efficace.
- **Nombre de poses.** Avec ±1 pixel et des décalages connus, 50 poses ramènent les trois bandes près de la qualité monochrome ; 16 poses suffisent pour G, pas pour R et B (B surtout, le plus sous-échantillonné).
- **Estimation des décalages.** Sur la mosaïque, elle plafonne vers 0,08 pixel du détecteur, contre environ 0,003 pixel en monochrome : le mélange des couleurs et les différences de PSF entre bandes biaisent le centroïde. Cette erreur multiplie l'erreur des PSF par 2 à 3. Ni une fenêtre plus large, ni des centroïdes par bande, ni le réajustement sur le modèle reconstruit (`pose_refine`) ne l'améliorent. Des décalages connus par ailleurs (platine pilotée, codeur, métrologie), fournis par `-S`, lèvent cette limite.
- **Décalage chromatique latéral.** Il est retrouvé à 0,06 pixel HR pour R − G dans le cas favorable (50 poses, ±1 pixel), mais seulement à 0,2–0,4 pixel HR dans les autres cas, et pour B − G.
- **Pas de repli au pas du détecteur.** Reconstruire au pas du détecteur (`-r 1`) ne contourne pas un dithering insuffisant : une PSF repliée ne se représente pas correctement à ce pas, et même la reconstruction monochrome y est moins bonne (erreur relative de 0,15 à 0,18).

## Estimation des décalages sur la mosaïque

Cas visé : 100 poses, décalages inconnus de plusieurs pixels au total (simulés uniformes dans ±2,5 pixels), PSF rouge la plus dégradée (défocalisation de 0,3 onde), 30 dB ; 2 réalisations (`validation/bayer_shifts.py`, `validation/results/bayer_shifts.txt`). Erreur RMS des décalages, puis erreur relative / erreur de FWHM des PSF :

| Variante | Décalages | R | G | B |
| --- | --- | --- | --- | --- |
| Décalages vrais (référence) | 0 | 0,055 / 0,2 % ; 0,052 / 0,1 % | 0,038 / 0,1 % ; 0,035 / 0,2 % | 0,036 / 0,2 % ; 0,052 / 0,2 % |
| Pseudo-luminance seule | 0,125 ; 0,067 px | 0,116 / 1,4 % ; 0,118 / 2,4 % | 0,122 / 1,6 % ; 0,119 / 3,8 % | 0,306 / 16,9 % ; 0,136 / 12,2 % |
| + 2 réajustements (défaut ≥ 50 poses) | 0,082 ; 0,058 px | 0,084 / 3,2 % ; 0,098 / 0,6 % | 0,098 / 3,0 % ; 0,124 / 4,6 % | 0,232 / 8,9 % ; 0,109 / 6,6 % |

Les deux valeurs de chaque case correspondent aux deux réalisations. Avec des décalages exacts, les trois bandes sont aussi bonnes qu'en monochrome, y compris R malgré sa PSF dégradée : une PSF plus large est mieux échantillonnée. L'écart restant vient entièrement de l'estimation des décalages, et B, la bande la plus fine donc la plus sous-échantillonnée, y est la plus sensible.

**Pourquoi l'estimation plafonne.** Les PSF des trois bandes n'ont ni la même largeur ni la même position (décalage chromatique). Le centroïde de la mosaïque dépend donc des couleurs qui échantillonnent le cœur, c'est-à-dire de la position de l'étoile dans le motif 2 × 2 : 75 à 83 % de l'erreur initiale s'explique par cette position. Un tel biais systématique est en partie absorbé par la PSF reconstruite (toutes les poses à la même position ont la même erreur), ce qui limite l'alternance entre décalages et reconstruction. Les tests menés (`validation/results/bayer_bias_calibration.txt`) :

- réajustement sur les PSF vraies : 0,002 pixel, la méthode de réajustement est exacte ; la limite vient du modèle reconstruit ;
- réajustement sur des modèles moins régularisés (κ = 1 ou 2) ou sur un profil de Moffat lisse : sans gain (0,080 à 0,083 pixel) ;
- centroïdes par bande sur chaque sous-réseau : pires (0,11 à 0,55 pixel), le verrouillage sur la grille domine au pas de 2 pixels ;
- correction du biais par simulation de l'estimateur sur les PSF reconstruites : 0,125 → 0,073 pixel en 3 itérations (PSF 0,072 / 0,086 / 0,189), mais la boucle dérive ensuite et aucun critère sans vérité terrain ne dit quand l'arrêter ; non intégrée.

## Contrôle d'une reconstruction Bayer

`sprite-psf-check --bayer RGGB` applique les contrôles monochromes à chaque bande, avec deux adaptations :

- **Poses entières** : les blocs de la prédiction croisée et les moitiés sont formés de poses ; tous les sous-réseaux d'une pose (les deux sous-images G comprises) sont mis de côté ensemble, comme une pose réelle. Les indicateurs par image sont rapportés par pose (moyenne des deux sous-réseaux pour G).
- **Stabilité chromatique** : les décalages R − G et B − G sont mesurés sur chacune des deux moitiés ; leur écart est l'erreur due au bruit sur le décalage chromatique.

Les décalages entre poses et les flux sont ceux de la reconstruction complète (réajustements compris), imposés aux reconstructions partielles. Le contrôle compte donc 1 + 3 × 7 reconstructions (5 blocs et 2 moitiés par bande), réparties sur `-j` processus : environ 30 s pour 50 poses de 84 × 84 pixels avec des décalages connus, 65 s avec des décalages estimés et 2 min pour 100 poses (10 cœurs). `--scan` ne traite pas les mosaïques.

Calibration sur mosaïques simulées (`validation/bayer_check_calibration.py`, `validation/results/bayer_check_calibration.txt` ; simulation de « Résultats sur données simulées », poses à ±1 pixel, 30 dB, κ = 4, `-N`, une réalisation). Rapport prédiction / ajustement, écart relatif entre moitiés et dispersion des flux par bande R / G / B, puis erreur relative vraie :

| Cas | Rapport | Écart entre moitiés | Dispersion des flux | Erreur vraie | Alertes |
| --- | --- | --- | --- | --- | --- |
| 50 poses, décalages connus | 1,03 / 1,01 / 1,04 | 0,09 / 0,01 / 0,09 | 0,5 / 1,1 / 0,7 % | 0,06 / 0,04 / 0,05 | 0 |
| 50 poses, décalages estimés | 1,04 / 1,03 / 1,09 | 0,16 / 0,03 / 0,12 | 2,6 / 7,2 / 4,8 % | 0,09 / 0,08 / 0,15 | 3 |
| 100 poses, décalages estimés | 1,02 / 1,01 / 1,04 | 0,03 / 0,02 / 0,04 | 2,2 / 7,7 / 4,0 % | 0,08 / 0,06 / 0,13 | 4 |
| 50 poses, sans réajustement des décalages | 1,03 / 1,09 / 1,11 | 0,14 / 0,08 / 0,16 | 1,9 / 17,2 / 4,2 % | 0,12 / 0,09 / 0,18 | 5 |
| 50 poses, flux ±20 % non estimés | 1,20 / 1,19 / 1,23 | 0,26 / 0,07 / 0,14 | 8,5 / 13,2 / 11,7 % | 0,11 / 0,14 / 0,12 | 10 |
| 50 poses, κ = 1 | 1,25 / 1,14 / 1,31 | 0,17 / 0,05 / 0,12 | 2,5 / 6,8 / 4,7 % | 0,30 / 0,23 / 0,32 | 6 |
| 16 poses, décalages connus | 4,76 / 1,07 / 3,15 | 0,47 / 0,15 / 0,39 | 0,6 / 1,5 / 2,9 % | 0,14 / 0,07 / 0,37 | 3 |

Lecture, en plus des règles du cas monochrome :

- **Seuils** : ceux du monochrome (χ²/N, rapport ≤ 1,1) restent valables. Sans défaut, aucune alerte ; flux variables et régularisation trop faible sont détectés dans les trois bandes.
- **Dispersion des flux et décalages** : avec des flux égaux, elle passe de 1 % (décalages exacts) à 4 à 8 % (décalages estimés) puis 17 % pour G sans réajustement : sur une mosaïque, les flux ajustés absorbent l'erreur des décalages. Une dispersion de quelques pour cent est donc attendue en Bayer et ne signifie pas forcément des flux variables ; le rapport le précise. Si les flux sont connus égaux, cette dispersion est un indicateur indirect de la qualité des décalages, utile pour comparer des réglages.
- **Moins de 32 poses** : chaque moitié n'a que 8 à 15 images R et B pour un sur-échantillonnage de 4 sur le sous-réseau ; les reconstructions partielles sont sous-déterminées et les indicateurs 2 et 3 de R et B très pessimistes (rapport 3 à 5 avec les décalages exacts). Le rapport le signale ; seul G reste interprétable.
- **Décalage chromatique** : l'écart entre moitiés (0,01 à 0,11 pixel HR avec 50 à 100 poses) ne voit pas l'erreur due aux décalages estimés, commune à toutes les poses : l'erreur vraie sur B − G reste vers 0,15 pixel HR avec des décalages estimés (0,006 avec des décalages connus). Un décalage chromatique mesuré n'est significatif qu'au-delà d'environ 0,2 pixel HR, sauf décalages connus par ailleurs.
- **Erreur des décalages** : comme en monochrome, le contrôle ne la mesure pas directement. Elle se manifeste par un rapport un peu plus élevé en B (1,04 à 1,11) et par la dispersion des flux.

## Pistes pour la mosaïque

1. **Vérifier l'amplitude réelle des décalages** sur les données (`pose_shift.txt`, en pixels du détecteur) : ils doivent couvrir au moins 2 pixels dans chaque direction.
2. **Autres étoiles dans le champ** : si les poses contiennent d'autres étoiles, leurs décalages sont les mêmes, mais à des positions différentes dans le motif ; moyenner leurs estimations réduirait le biais lié à la position, qui ne se moyenne pas sur une seule étoile.
3. **Critère d'arrêt pour la correction du biais** (par exemple la prédiction croisée), qui la rendrait utilisable.
4. **Décalages connus** : si le déplacement entre poses est mesuré par ailleurs, le fournir par `-S` lève la limite d'estimation.
5. **Reconstruction couplée des bandes** (écartée pour le cas visé, les PSF des bandes étant trop différentes) : parcimonie de groupe sur les coefficients d'ondelettes des trois bandes (`GroupLASSO` dans ModOpt), ou terme liant les bandes après remise à l'échelle chromatique.
6. **Modèle optique commun** : un front d'onde unique propagé à la longueur d'onde de chaque bande, les aberrations chromatiques devenant des paramètres physiques. C'est un problème de reconstruction de phase plutôt qu'une extension de SPRITE.
