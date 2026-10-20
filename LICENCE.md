# Licences

Ce dépôt réunit deux codes sous des licences différentes.

## Code C++ d'origine (`sprite_c/`) — licence CeCILL v2.0

SPRITE v1.0 (*SParse Recovery of InsTrumental rEsponse*), CEA, octobre 2014, de F. M. Ngolè Mboula et J.-L. Starck (laboratoire AIM, CEA Saclay / CosmoStat). Il est distribué sous la licence CeCILL v2.0 du 05/09/2006 ; le texte intégral se trouve dans [`sprite_c/LICENSE.CeCILL`](sprite_c/LICENSE.CeCILL). Le code est reproduit ici sans modification.

- Code publié : http://www.cosmostat.org/software/sprite (page indiquée dans l'article : http://www.cosmostat.org/sprite.html)
- Référence logicielle : Astrophysics Source Code Library, [ascl:1506.008](https://ascl.net/1506.008)

Les fichiers de `reference/` (script de compilation, programme d'export `dump_ref.cc` lié aux bibliothèques du C++, données de référence produites par le binaire d'origine) dérivent de ce code et relèvent de la même licence CeCILL v2.0.

## Publication

F. M. Ngolè Mboula, J.-L. Starck, S. Ronayette, K. Okumura, J. Amiaux, *Super-resolution method using sparse regularization for point-spread function recovery*, Astronomy & Astrophysics 575, A86, 2015, DOI [10.1051/0004-6361/201424167](https://doi.org/10.1051/0004-6361/201424167) ; prépublication [arXiv:1410.7679](https://arxiv.org/abs/1410.7679). L'article n'est pas redistribué dans ce dépôt.

## Code Python (`sprite_py/`, `docs/`) — licence CeCILL v2.1

Copyright (c) 2026 Olivier Amram

Le portage Python est une œuvre dérivée du code C++ ci-dessus (traduction de ses algorithmes et de ses conventions). Il est distribué, comme lui, sous licence CeCILL, dans sa version 2.1 du 21/06/2013, conformément aux articles 5.3.2 (logiciel modifié soumis à la licence) et 12.3 (redistribution sous une version ultérieure) de la CeCILL v2.0 ; le texte intégral se trouve dans [`sprite_py/LICENSE.CeCILL-2.1`](sprite_py/LICENSE.CeCILL-2.1) et sur https://cecill.info.

Ce portage est indépendant et non officiel : il n'est ni développé, ni maintenu, ni approuvé par le CEA ou l'équipe CosmoStat.

En résumé : chacun peut utiliser, étudier, modifier et redistribuer ce code ; toute version modifiée redistribuée doit l'être sous la même licence, avec son code source. La CeCILL v2.1 est compatible avec la GNU GPL v2 et v3, et relève du droit français.
