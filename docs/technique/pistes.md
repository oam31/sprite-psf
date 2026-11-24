# Pistes


Ce qui change d'une version à l'autre est dans `CHANGELOG.md`.

## Par ordre de priorité

Par ordre de priorité proposé :

1. **Valider sur les images réelles** avec `sprite-psf-check`, en suivant l'ordre de réglage de « [Converger vers le meilleur réglage](donnees-reelles.md#4-converger-vers-le-meilleur-réglage) ». L'efficacité des contrôles n'est établie que sur simulations, en monochrome comme en Bayer.
2. **Mosaïque de Bayer** : améliorer l'estimation des décalages (autres étoiles du champ, critère d'arrêt pour la correction du biais, que la prédiction croisée de `sprite-psf-check --bayer` pourrait fournir) ; balayage de paramètre (`--scan`) en mode Bayer ; voir « [Mosaïque de Bayer](bayer.md) ».
3. **Outil d'injection** : ajouter une PSF simulée connue à des images réelles sans étoile, avec les décalages et flux réels, pour mesurer l'erreur vraie avec le bruit et les défauts réels du détecteur.
4. **Simulations plus réalistes** : réponse du pixel, bruit de photons de l'étoile, SNR plus faibles (5 à 10 dB), pour confirmer les seuils de ce document.
5. **Facteur de sur-échantillonnage 3** avec de nombreuses poses : à tester sur simulations.
6. **Estimation des flux dans l'optimisation** (mise à jour fermée à chaque itération), plus économique que les réajustements, si les flux varient dans les données réelles.
7. **Fond non uniforme** : ajuster un plan sur la bordure, ou estimer le fond conjointement à la PSF.
8. **Options non portées de `run_sprite`** : FISTA (`-A 1`), simulation du bruit (`-e 1`), autres premières estimations (`-M`), déconvolution, pondérations spatiales (`-V`), seulement en cas de besoin.

## Hors périmètre : variation spatiale de la PSF

Cette évolution est hors périmètre : le cas d'usage visé dispose de nombreuses poses de la même étoile, donc d'une même PSF, ce qui est exactement le cadre de SPRITE. L'analyse est conservée pour mémoire.

Sur un grand champ, la PSF varie avec la position, et chaque étoile n'est souvent observée qu'une fois. L'information sous-pixel vient alors de la position aléatoire de nombreuses étoiles par rapport à la grille, et il faut un modèle de PSF sur tout le champ, x(u, v) :

| | SPRITE | Modèle de champ |
| --- | --- | --- |
| Données | n poses d'une étoile | N étoiles, une ou quelques poses chacune |
| Inconnue | une PSF HR | une PSF HR en tout point du champ |
| Diversité sous-pixel | décalages entre poses | positions des étoiles sur la grille |
| Sortie | une image | un modèle évaluable en (u, v) |

Approches publiées, de la plus proche du code actuel à la plus éloignée :

1. **SPRITE par position puis interpolation** des PSF HR sur le champ ; demande au moins 4 poses décalées par étoile et ne partage pas l'information entre étoiles.
2. **Modèle polynomial conjoint régularisé par parcimonie**, à la PSFEx : x(u, v) = Σ<sub>j</sub> p<sub>j</sub>(u, v) S<sub>j</sub>, monômes de degré ≤ d. Le problème reste linéaire et convexe ; la positivité aux positions des étoiles se traite par l'algorithme primal-dual de Condat-Vũ (ModOpt). C'est l'extension directe de SPRITE.
3. **RCA** (Ngolè Mboula et al.) : factorisation de rang faible des PSF du champ, composantes parcimonieuses en ondelettes et coefficients réguliers sur un graphe des positions ; minimisation alternée.
4. **MCCD** (Liaudat et al.) : RCA étendu à un plan focal en mosaïque, composantes globales et locales par détecteur.
5. **Modèles optiques différentiables**, par exemple WaveDiff (Liaudat et al.) : front d'onde paramétré sur le champ, propagation optique et correction apprise.
