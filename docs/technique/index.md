# Documents techniques

Document technique du portage Python de SPRITE, découpé par thème. Ces
documents décrivent l'état présent du code ; ce qui change d'une version à
l'autre est dans les [notes de version](../changelog.md).

| Document | Contenu |
|---|---|
| [Objet et méthode publiée](methode.md) | Objet et périmètre du portage, organisation du dépôt, modèle d'observation, a priori parcimonieux, dictionnaires, algorithme |
| [Implémentation et contrôle du portage](implementation.md) | Implémentation sur ModOpt, choix d'implémentation, référence C++, écarts mesurés, validation, erreurs du C++ corrigées |
| [Utilisation et paramètres](utilisation.md) | Installation, API, ligne de commande, formats ; paramètres ; fond, pixels saturés ou défectueux |
| [Décalages, prox, flux et nombre d'images](etudes.md) | Études : estimation des décalages, adjoint exact du prox d'analyse, flux variables, nombre d'images |
| [Mosaïque de Bayer](bayer.md) | Reconstruction par bande, décalages sur la mosaïque, contrôle, pistes |
| [Mise en œuvre sur données réelles](donnees-reelles.md) | Préparer, reconstruire, contrôler, converger vers le meilleur réglage, limites de la validation |
| [Pistes](pistes.md) | Par ordre de priorité ; variation spatiale de la PSF (hors périmètre) |
| [Références](bibliographie.md) | Références |

```{toctree}
:hidden:
:maxdepth: 2

methode
implementation
utilisation
etudes
bayer
donnees-reelles
pistes
bibliographie
```
