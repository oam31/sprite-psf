# Historique

Évolutions du portage Python de SPRITE, de la plus récente à la plus ancienne. Le détail des méthodes, des mesures et des paramètres est dans [`docs/document_technique.md`](docs/document_technique.md) ; les évolutions à venir y sont listées dans la partie « Évolutions ».

## Version 1.0 — 20/10/2026

Première version publiée. L'historique git du développement n'a pas été conservé ; les étapes sont résumées ci-dessous.

| Date | Évolution | Effet |
| --- | --- | --- |
| 20/10/2026 | Version 1.0 ; licences (C++ CeCILL v2.0 d'origine, portage Python CeCILL v2.1) ; historique séparé du document technique | |
| 19/10/2026 | `sprite-psf-check --bayer` : contrôles par bande sur poses entières, stabilité chromatique ; calibration sur simulation | contrôle de cohérence des mosaïques |
| 19/10/2026 | Reconstructions indépendantes en parallèle (`-j`) : bandes, blocs, moitiés, balayages | contrôle 1,9 fois plus rapide (50 poses, 10 cœurs) |
| 19/10/2026 | Dossier du C++ renommé `sprite_c` ; couverture des tests portée à 97 % (68 tests) | dépôt plus lisible ; chemins d'utilisation tous testés |
| 18/10/2026 | Bayer : réajustement des décalages sur les PSF reconstruites à partir de 50 poses ; étude du biais de l'estimateur | erreur des décalages réduite d'environ un tiers |
| 18/10/2026 | Reconstruction par bande d'images Bayer brutes (`--bayer RGGB`), diagnostic chromatique | trois PSF sur une grille commune ; exige des décalages d'au moins ±1 pixel |
| 18/10/2026 | Défauts : starlet et κ = 4 | réglages mesurés comme les meilleurs |
| 18/10/2026 | Balayage de paramètre par validation croisée (`sprite-psf-check --scan`), règle « une erreur type » | κ retenu à moins de 7 % de l'optimum sur simulations |
| 18/10/2026 | Commande `sprite-psf-check` : ajustement, prédiction croisée, moitiés, dispersion des flux | contrôle de cohérence sur données réelles |
| 18/10/2026 | Options regroupées par thème ; rétrocompatibilité C++ isolée, avec une option par erreur | paramétrage lisible |
| 18/10/2026 | Mesure de la variation de flux tolérable | règle : estimer les flux au-delà de 3 % |
| 17/10/2026 | Flux par ajustement, réajustements, flux imposés (`--flux-method fit`, `--flux-refine`, `--fluxes`) | rattrape des variations de ±50 % |
| 17/10/2026 | Adjoint exact dans le prox d'analyse pour la starlet ; transformées 5 fois plus rapides | plus de fond étendu ; erreur de la starlet divisée par 2 à 4 |
| 17/10/2026 | Pixels défectueux exclus, pixels saturés censurés (`--bad-pixels`, `--saturation`) | pixels chauds neutralisés ; saturation mieux traitée qu'exclue |
| 17/10/2026 | Annotations de types et docstrings numpy en français | code documenté |
| 17/10/2026 | `centroid_wide` devient l'estimateur de décalages par défaut | erreur relative divisée par 2 à 3 de 20 à 40 dB |
| 16/10/2026 | Options d'estimation des décalages : `centroid_wide`, `xcorr`, `shift_refine` | voir « Estimation des décalages » |
| 16/10/2026 | Entrées TIFF 16 bits et float32 ; soustraction du fond de bord | données de laboratoire utilisables directement |
| 16/10/2026 | Corrections du recalage de la première estimation, de l'ouverture de flux et du critère d'arrêt | voir « Erreurs du C++ corrigées » dans le document technique |
| 15/10/2026 | Portage Python sur ModOpt, validé contre le C++ recompilé | écart de 10⁻⁶ au C++ |
