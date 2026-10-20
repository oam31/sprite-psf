# SPRITE — super-résolution de PSF

Portage Python, corrigé et étendu, de SPRITE (Ngolè Mboula et al., A&A 2015,
arXiv:1410.7679), méthode du CEA de super-résolution de PSF à partir de poses
décalées d'une même étoile.

> Portage indépendant et non officiel : il n'est ni développé, ni maintenu, ni
> approuvé par le CEA ou l'équipe CosmoStat. Le SPRITE d'origine est publié sur
> http://www.cosmostat.org/software/sprite.

Article : F. M. Ngolè Mboula et al., A&A 575, A86 (2015),
[DOI 10.1051/0004-6361/201424167](https://doi.org/10.1051/0004-6361/201424167),
[arXiv:1410.7679](https://arxiv.org/abs/1410.7679).

| Dossier | Rôle |
| --- | --- |
| [`sprite_py/`](sprite_py/) | le logiciel : paquet Python `sprite-psf`, tests, validations |
| [`docs/`](docs/document_technique.md) | document technique : méthode, paramètres, réglage, résultats |
| [`reference/`](reference/) | oracle des tests : C++ d'origine recompilé et ses sorties de référence |
| [`sprite_c/`](sprite_c/) | code C++ SPRITE v1.0 d'origine, non modifié |
| [`HISTORIQUE.md`](HISTORIQUE.md) | évolutions traitées, par version |
| [`LICENCE.md`](LICENCE.md) | licences : C++ sous CeCILL v2.0, Python sous CeCILL v2.1 ; références de la publication et du code C++ |

Installation directe depuis GitHub (le paquet est dans le sous-dossier `sprite_py`) :

```bash
pip install "git+https://github.com/oam31/sprite-psf.git#subdirectory=sprite_py"
```

Démarrage rapide depuis une copie locale :

```bash
cd sprite_py
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/sprite-psf -N --background images/*.tif psf.tif sorties/
.venv/bin/sprite-psf-check -N --background images/*.tif
```

Version 1.0. Licences : voir [`LICENCE.md`](LICENCE.md).
