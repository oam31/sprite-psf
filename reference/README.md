# Référence C++ SPRITE v1.0

Ce dossier sert d'oracle pour le portage Python (`../sprite_py`). Les sources
d'origine (`../sprite_c`) ne sont jamais modifiées.

## Reconstruire

```bash
./build_reference.sh
```

Prérequis Homebrew : `cmake`, `cfitsio`, `gsl`, `libomp`. Le script :

1. copie `../sprite_c` dans `sprite_c_build/` et applique trois correctifs de
   compilation, sans effet sur les calculs (dialecte C++98 et libomp pour Apple
   clang, setters `std::complex` dans un lecteur FITS non utilisé, renommage de
   la globale `transform` en conflit avec `std::transform`) ;
2. compile `run_sprite` et le programme d'export `dump_ref.cc` ;
3. exécute les deux configurations du papier (section 6) sur `cube_test.fits` :
   `run_sprite -t {2,24} -s 4 -r 2 -F -N -W -w resi.fits`.

Les sorties sont déterministes : deux reconstructions complètes donnent des
fichiers identiques bit à bit (vérifié par MD5 sur les 103 fichiers).

## Contenu de `golden/`

| Dossier | Origine | Contenu |
|---|---|---|
| `t2/`, `t24/` | `run_sprite` | solution, première estimation, résidus, centroïdes, décalages, flux, bruit, traces |
| `units_t2/`, `units_t24/` | `dump_ref` | sorties intermédiaires : transformées et reconstructions, convolution FFT, décimation, opérateurs M et Mᵀ sur entrées aléatoires, prox d'analyse, pile de Lanczos, filtrage par support multirésolution, co-addition et ajustement gaussien de `lr_psf_size`, rayon spectral |

## Conventions d'orientation

- Image 2D iSAP (`Ifloat(i,j)`) écrite en FITS → `numpy[i, j]`.
- Cube iSAP (`fltarray(x,y,z)`) → `numpy[z, y, x]`.
- Le C++ copie `Ifloat(i,j) = fltarray(i,j,k)` : il travaille sur les images
  **transposées**. Ses sorties 2D (`solution.fits`, `model.fits`) sont donc la
  transposée de l'orientation du cube d'entrée ; les cubes (`lr_solution.fits`,
  `resi.fits`) ne le sont pas.
- Fichiers texte : colonne 1 = axe x (dernier axe numpy), colonne 2 = axe y.
