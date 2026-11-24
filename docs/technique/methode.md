# Objet et méthode publiée

## Objet et périmètre

Ce document décrit SPRITE, une méthode de super-résolution de PSF publiée par le CEA en 2014, et son portage en Python (paquet `sprite-psf`). Le portage reproduit le code C++ d'origine à la précision float32 près, puis l'améliore :

- il corrige quatre erreurs du code d'origine ;
- il remplace l'estimateur de décalages du C++ par un estimateur sans biais de grille ;
- il traite les pixels saturés ou défectueux ;
- il estime les flux par ajustement quand ils varient entre poses ;
- il lit les images TIFF 16 bits ou float32 ;
- il fournit un outil de contrôle des reconstructions sur données réelles ;
- il reconstruit une PSF par bande à partir d'images brutes en mosaïque de Bayer (RGGB).

SPRITE (*SParse Recovery of InsTrumental rEsponse*) reconstruit une PSF à haute résolution (HR) à partir de plusieurs images à basse résolution (BR) d'une même source ponctuelle, décalées entre elles d'une fraction de pixel. Le cas d'usage d'origine est la mission Euclid, dont le détecteur échantillonne l'optique à environ 0,688 fois la fréquence de Nyquist : les étoiles y sont repliées (aliasées), alors que la mesure de cisaillement gravitationnel exige une PSF connue à mieux que quelques pour cent. Le cas d'usage visé ici dispose de nombreuses poses (jusqu'à 100) de la même étoile.

Périmètre :

- **Méthode** : algorithme 2 du papier, avec un a priori parcimonieux en analyse, une contrainte de positivité et une repondération.
- **Code de référence** : SPRITE v1.0 (CEA, octobre 2014, licence CeCILL), exécutable `run_sprite`.
- **Portage** : Python 3, numpy, scipy, astropy, tifffile, et ModOpt pour l'algorithme d'optimisation.
- **Hors périmètre** : la variation spatiale de la PSF (voir la dernière partie, pour mémoire) et les variantes secondaires de `run_sprite` (FISTA, simulation du bruit, autres premières estimations, déconvolution, pondérations spatiales).

## Organisation du dépôt

| Dossier ou fichier | Contenu | Rôle |
| --- | --- | --- |
| `sprite_c/` | code C++ SPRITE v1.0 d'origine (CEA, CeCILL), jamais modifié | archive du code publié |
| `reference/` | script de compilation du C++, programme d'export `dump_ref.cc`, données de référence `golden/` | oracle des tests : sorties du binaire d'origine, finales et intermédiaires |
| `sprite_py/` | paquet Python `sprite-psf` : code, tests, scripts de validation et leurs résultats | le logiciel à utiliser |
| `docs/` | documents techniques (`docs/technique/`) et moteur de la documentation (Sphinx, MyST) | documentation |
| `CHANGELOG.md` | ce qui change d'une version à l'autre | notes de version |
| `LICENCE.md` | licences (C++ CeCILL v2.0, Python CeCILL v2.1), références de la publication et du code publié | licence |

Le dossier `reference/` sert à prouver que le portage est fidèle. `build_reference.sh` recompile le C++ d'origine dans une copie (`sprite_c_build/`, non versionnée), l'exécute sur `cube_test.fits` avec les réglages du papier, et exporte ses résultats intermédiaires. Les fichiers de `golden/` (22 Mo, versionnés) sont ensuite comparés au Python par les tests. La reconstruction est reproductible bit à bit : on peut régénérer `golden/` à tout moment et vérifier qu'il n'a pas changé. Ce dossier n'est pas nécessaire pour utiliser `sprite-psf`, seulement pour exécuter les tests de fidélité au C++ (ils sont ignorés s'il est absent).

## Méthode publiée

SPRITE résout un problème inverse linéaire régularisé par la parcimonie de la correction à apporter à une première estimation, dans un dictionnaire d'ondelettes redondant, avec positivité et repondération (Ngolè Mboula et al. 2015, sections 2 et 3).

### Modèle d'observation

On dispose de n images BR y<sub>k</sub> de taille p₁ × p₂ d'une même étoile. La PSF recherchée x est échantillonnée d fois plus finement sur chaque axe (d = 2 par défaut). Chaque image est la PSF décalée d'une fraction de pixel, puis décimée, avec un flux et un bruit propres :

```math
y_k = f_k \, D \, H_k \, x + n_k, \qquad k = 1 \dots n
```

- **H<sub>k</sub>** : convolution par un noyau d'interpolation 2D séparable centré sur le décalage de l'image k. Le papier utilise un noyau de Lanczos, L(x) = sinc(x) sinc(x/a).
- **D** : décimation, qui garde un pixel HR sur d dans chaque direction.
- **f<sub>k</sub>** : flux photométrique relatif de l'image k (temps de pose, transparence, stabilité de la source).
- **n<sub>k</sub>** : bruit blanc gaussien d'écart type σ<sub>k</sub>.

En empilant les images blanchies (divisées par leur σ<sub>k</sub>), le terme d'attache aux données s'écrit :

```math
J_1(x) = \frac{1}{2} \sum_{k=1}^{n} \frac{\lVert y_k - f_k D H_k x \rVert_2^2}{\sigma_k^2} = \frac{1}{2} \lVert z - M x \rVert_2^2
```

M est de taille n p₁ p₂ × d² p₁ p₂. Avec n = 4 et d = 2, il y a autant d'équations que d'inconnues, mais le système est mal conditionné et bruité : il faut un a priori. La super-résolution n'est possible que si les décalages couvrent des phases sous-pixel variées : des poses toutes à la même phase ne contiennent pas plus d'information qu'une seule.

### A priori parcimonieux en analyse

La solution s'écrit x = x⁽⁰⁾ + Δ, où x⁽⁰⁾ est une première estimation. C'est la correction Δ qui est supposée parcimonieuse dans un dictionnaire Φ d'ondelettes non décimées. Le papier choisit la forme « analyse », qui pénalise directement les coefficients ΦΔ, plutôt que la forme « synthèse », qui suppose Δ combinaison de peu d'atomes. Avec la positivité de la PSF :

```math
\min_{\Delta} \; J_1\left(x^{(0)} + \Delta\right) + \kappa \left\lVert w^{(k)} \odot \lambda \odot \Phi \Delta \right\rVert_1 + \iota_{\{\Delta \ge -x^{(0)}\}}(\Delta)
```

- **λ** : niveaux de bruit attendus sur chaque coefficient. Ils sont réestimés à chaque itération par le MAD, bande par bande, de la transformée du pas de gradient.
- **κ** : seuil en nombre de « sigmas » ; c'est le principal réglage de la méthode (4 dans les expériences du papier).
- **w⁽ᵏ⁾** : poids de repondération.
- **ι** : indicatrice de la contrainte Δ ≥ −x⁽⁰⁾, c'est-à-dire x ≥ 0.

### Dictionnaires d'ondelettes : starlet et 7/9

Une transformée en ondelettes décompose une image en couches de détails à des échelles successives (1, 2, 4, 8… pixels), plus une image lissée, l'échelle grossière. Une PSF s'y décrit avec peu de coefficients significatifs : un cœur compact aux petites échelles, des anneaux et des ailes aux échelles moyennes. C'est cette parcimonie que SPRITE exploite pour séparer la PSF du bruit.

Les deux dictionnaires de SPRITE sont **non décimés** : chaque bande garde la taille de l'image. Ils sont donc redondants (plus de coefficients que de pixels) et invariants par translation : décaler l'image d'un pixel décale ses coefficients d'un pixel, sans les modifier. C'est indispensable pour une PSF dont la position sous-pixel varie, et cela évite les artefacts de bloc des transformées décimées.

| | Starlet (`-t 2`) | 7/9 non décimée (`-t 24`) |
| --- | --- | --- |
| Construction | algorithme « à trous » : lissages successifs par une B-spline cubique séparable, filtre (1, 4, 6, 4, 1)/16 dilaté d'un facteur 2 à chaque échelle | banc de filtres séparables bi-orthogonaux d'Antonini (7 et 9 coefficients), appliqués aux lignes puis aux colonnes, dilatés à chaque échelle |
| Bandes par échelle | 1, isotrope : w<sub>j</sub> = c<sub>j</sub> − c<sub>j+1</sub>, différence de deux lissages | 3, orientées : horizontale, verticale, diagonale |
| Allure des atomes | taches rondes, positives au centre, entourées d'un anneau négatif | motifs orientés selon les axes de la grille |
| Nombre de bandes (image 168 × 168, 6 échelles) | 6 | 16 |
| ‖Φ‖² (redondance effective) | 1,33 | 1,81 |
| Reconstruction | somme des bandes | banc de synthèse (filtres différents de ceux d'analyse) |
| Adjoint | différent de la reconstruction | différent de la reconstruction |
| Usage typique | objets isotropes et positifs : étoiles, galaxies (conçue pour l'astronomie) | contours et structures orientées |
| Temps (4 images 84 × 84) | 2 s | 6 s |

Pour une PSF, la starlet décrit naturellement le cœur et les anneaux ; la 7/9 décrit mieux les structures alignées sur la grille (aigrettes de diffraction d'une araignée à branches verticales ou horizontales, par exemple), mais produit des artefacts en croix sur les structures rondes. Le C++ calculait mal le prox d'analyse de la starlet (voir « [Prox d'analyse : adjoint exact](etudes.md#prox-danalyse--adjoint-exact) »). Une fois corrigée, la starlet égale la 7/9 de 20 à 40 dB, fait mieux à faible SNR et tourne 3 fois plus vite : c'est le dictionnaire par défaut depuis le 18 octobre 2026.

### Repondération

La norme ℓ₁ biaise les gros coefficients. SPRITE applique la repondération de Candès, Wakin et Boyd (2008) : après chaque résolution, les poids deviennent

```math
w_j^{(k+1)} = \frac{1}{1 + \lvert \alpha_j^{(k)} \rvert / (3 \sigma_j)}, \qquad \alpha^{(k)} = \Phi \Delta^{(k)}
```

Les coefficients significatifs sont ainsi moins pénalisés à la passe suivante. Par défaut, une passe initiale est suivie d'une passe repondérée.

### Algorithme : forward-backward généralisé

Le problème est la somme d'un terme différentiable (J₁) et de deux termes non lisses (ℓ₁ pondérée, positivité). SPRITE le résout par le forward-backward généralisé de Raguet, Fadili et Peyré (2013), avec deux variables auxiliaires z₁ et z₂ et des poids ω₁ = ω₂ = 0,5 :

```math
\begin{aligned}
z_i &\leftarrow z_i + \lambda \left( \operatorname{prox}_{\frac{\mu}{\omega_i} g_i}\!\left( 2\Delta - z_i - \mu \nabla J_1(x^{(0)} + \Delta) \right) - \Delta \right), \quad i = 1, 2 \\
\Delta &\leftarrow \omega_1 z_1 + \omega_2 z_2
\end{aligned}
```

- **Pas** : μ = 1/ρ(MᵀM), le rayon spectral étant estimé par la méthode de la puissance.
- **Relaxation** : λ = 1,4.
- **Prox de la positivité** : projection Δ ← max(Δ, −x⁽⁰⁾).
- **Prox de l'ℓ₁ d'analyse** : il n'a pas de forme close, car Φ est redondant. Il se calcule par un FISTA sur le problème dual (seuillage doux, puis projection sur une boule ℓ∞ pondérée), qui utilise l'adjoint Φᵀ : 100 sous-itérations au premier appel, puis 5 avec démarrage à chaud.

### Estimation des paramètres

Tout est estimé automatiquement à partir des images BR (section 3.4.1 du papier) :

- **Bruit σ<sub>k</sub>** : MAD de la première bande de détails d'une transformée 7/9, divisé par 0,6745.
- **Décalages** : différences de centroïdes entre chaque image et l'image de référence, calculés sur les images seuillées à min(4σ<sub>k</sub>, (max|y<sub>k</sub>|/σ<sub>k</sub> − 1)σ<sub>k</sub>), avec la pondération gaussienne itérative de Baker et Moallem (2007).
- **Flux** : intégrale dans une ouverture circulaire centrée sur le centroïde.
- **Première estimation x⁽⁰⁾** : shift-and-add des images interpolées par Lanczos (médiane), puis débruitage par seuillage dur à 5σ par échelle (section 3.4.2). Comme x⁽⁰⁾ n'est pas régularisée, ses défauts passent dans la solution : elle doit être propre.

Coût annoncé dans le papier : O(n p₁ p₂ d² log(p₁ p₂ d²)), soit environ 60 s pour quatre images 84 × 84 en 2014.
