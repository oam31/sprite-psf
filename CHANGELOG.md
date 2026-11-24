# Change Log

## Version History

Non publié

:   -   Document technique découpé par thème dans `docs/technique/`
        (méthode, implémentation, utilisation, études, mosaïque de
        Bayer, données réelles, pistes, références).
    -   Documentation réorganisée : document technique dans
        `docs/technique/`, publié avec l'API et les notes de version par
        Sphinx et MyST (`docs/`, extra `docs` de `sprite_py`) ; ce fichier
        remplace `HISTORIQUE.md`.

1.0.0 (2026-10-20)

:   -   Première version publiée : portage Python de SPRITE sur ModOpt,
        validé contre le C++ recompilé (écart de 10⁻⁶) ; licences : C++
        CeCILL v2.0 d'origine, portage Python CeCILL v2.1.
    -   `sprite-psf-check --bayer` : contrôles par bande sur poses
        entières, stabilité chromatique ; calibration sur simulation :
        contrôle de cohérence des mosaïques.
    -   Reconstructions indépendantes en parallèle (`-j`) : bandes, blocs,
        moitiés, balayages : contrôle 1,9 fois plus rapide (50 poses, 10
        cœurs).
    -   Dossier du C++ renommé `sprite_c` ; couverture des tests portée à
        97 % (68 tests) : dépôt plus lisible ; chemins d'utilisation tous
        testés.
    -   Bayer : réajustement des décalages sur les PSF reconstruites à
        partir de 50 poses ; étude du biais de l'estimateur : erreur des
        décalages réduite d'environ un tiers.
    -   Reconstruction par bande d'images Bayer brutes (`--bayer RGGB`),
        diagnostic chromatique : trois PSF sur une grille commune ; exige
        des décalages d'au moins ±1 pixel.
    -   Défauts : starlet et κ = 4 : réglages mesurés comme les meilleurs.
    -   Balayage de paramètre par validation croisée (`sprite-psf-check
        --scan`), règle « une erreur type » : κ retenu à moins de 7 % de
        l'optimum sur simulations.
    -   Commande `sprite-psf-check` : ajustement, prédiction croisée,
        moitiés, dispersion des flux : contrôle de cohérence sur données
        réelles.
    -   Options regroupées par thème ; rétrocompatibilité C++ isolée, avec
        une option par erreur : paramétrage lisible.
    -   Mesure de la variation de flux tolérable : règle : estimer les
        flux au-delà de 3 %.
    -   Flux par ajustement, réajustements, flux imposés (`--flux-method
        fit`, `--flux-refine`, `--fluxes`) : rattrape des variations de
        ±50 %.
    -   Adjoint exact dans le prox d'analyse pour la starlet ;
        transformées 5 fois plus rapides : plus de fond étendu ; erreur de
        la starlet divisée par 2 à 4.
    -   Pixels défectueux exclus, pixels saturés censurés (`--bad-pixels`,
        `--saturation`) : pixels chauds neutralisés ; saturation mieux
        traitée qu'exclue.
    -   Annotations de types et docstrings numpy en français : code
        documenté.
    -   `centroid_wide` devient l'estimateur de décalages par défaut :
        erreur relative divisée par 2 à 3 de 20 à 40 dB.
    -   Options d'estimation des décalages : `centroid_wide`, `xcorr`,
        `shift_refine` : voir « Estimation des décalages ».
    -   Entrées TIFF 16 bits et float32 ; soustraction du fond de bord :
        données de laboratoire utilisables directement.
    -   Corrections du recalage de la première estimation, de l'ouverture
        de flux et du critère d'arrêt : voir « Erreurs du C++ corrigées »
        dans `docs/technique/implementation.md`.
