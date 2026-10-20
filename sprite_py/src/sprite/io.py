"""Lecture des images BR et écriture des résultats : cubes FITS et images TIFF.

Entrées acceptées (une image BR par plan ou par bande) :

* un cube FITS ``(n_frames, ny, nx)`` ;
* un TIFF multipage, ou un TIFF dont les échantillons ou canaux sont les bandes ;
* plusieurs fichiers TIFF ou FITS d'une image chacun (liste, dossier ou motif).

Les échantillons TIFF peuvent être des entiers non signés 16 bits ou des
flottants 32 bits ; les autres types entiers et flottants sont aussi acceptés.
Tout est converti en float64.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray

TIFF_EXT = {".tif", ".tiff"}
FITS_EXT = {".fits", ".fit", ".fts"}
_BAND_AXES = "SCQIZT"  # codes d'axes tifffile pouvant porter les images

PathLike = Union[str, Path]


def _read_tiff(path: Path) -> tuple[NDArray, np.dtype]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
    if data.ndim == 2:
        return data[None], data.dtype
    if data.ndim != 3:
        raise ValueError(f"{path} : image 2D ou 3D attendue, axes {axes!r} {data.shape}")
    band = next((i for i, a in enumerate(axes) if a in _BAND_AXES), 0)
    return np.moveaxis(data, band, 0), data.dtype


def _read_fits(path: Path) -> tuple[NDArray, np.dtype]:
    data = fits.getdata(path)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"{path} : image 2D ou cube 3D attendu, forme {data.shape}")
    return data, data.dtype


def _expand(inputs: PathLike | Iterable[PathLike]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in TIFF_EXT | FITS_EXT)
            if not found:
                raise FileNotFoundError(f"aucun fichier TIFF ou FITS dans {p}")
            paths.extend(found)
        elif any(c in str(item) for c in "*?["):
            found = sorted(Path(f) for f in glob.glob(str(item)))
            if not found:
                raise FileNotFoundError(f"aucun fichier ne correspond à {item}")
            paths.extend(found)
        else:
            paths.append(p)
    return paths


def _read_planes(inputs: PathLike | Iterable[PathLike]) -> tuple[list[NDArray], list[str]]:
    planes: list[NDArray] = []
    dtypes: list[str] = []
    for path in _expand(inputs):
        ext = path.suffix.lower()
        if ext in TIFF_EXT:
            data, dtype = _read_tiff(path)
        elif ext in FITS_EXT:
            data, dtype = _read_fits(path)
        else:
            raise ValueError(f"{path} : format non pris en charge (FITS ou TIFF)")
        planes.extend(data)
        dtypes.append(np.dtype(dtype).name)
    return planes, dtypes


def load_frames(inputs: PathLike | Iterable[PathLike]) -> tuple[NDArray[np.float64], list[str]]:
    """Charge les images BR.

    Parameters
    ----------
    inputs : str, Path or iterable of them
        Un chemin (cube FITS, TIFF multipage ou multibande, dossier, motif
        glob) ou une liste de chemins, lus dans l'ordre.

    Returns
    -------
    frames : ndarray, shape (n_frames, ny, nx)
        Images en float64.
    dtypes : list of str
        Type des échantillons de chaque fichier lu (``"uint16"``,
        ``"float32"``...).

    Raises
    ------
    ValueError
        Format non pris en charge, images de tailles différentes, ou moins de
        deux images.
    FileNotFoundError
        Dossier vide ou motif sans correspondance.
    """
    planes, dtypes = _read_planes(inputs)
    if len({p.shape for p in planes}) != 1:
        raise ValueError(f"images de tailles différentes : {sorted({p.shape for p in planes})}")
    if len(planes) < 2:
        raise ValueError("SPRITE demande au moins deux images")
    return np.array(planes, dtype=float), dtypes


def load_bad_pixels(inputs: PathLike | Iterable[PathLike]) -> NDArray[np.bool_]:
    """Charge une carte de pixels défectueux (FITS ou TIFF, non nul = défectueux).

    Parameters
    ----------
    inputs : str, Path or iterable of them
        Une carte commune à toutes les images (une seule image) ou une carte
        par image (cube, ou autant de fichiers que d'images).

    Returns
    -------
    ndarray of bool, shape (ny, nx) or (n_frames, ny, nx)
        ``True`` pour les pixels défectueux.
    """
    planes, _ = _read_planes(inputs)
    bad = np.array(planes) != 0
    return bad[0] if len(bad) == 1 else bad


def border_background(
    frames: NDArray[np.float64], width: int | None = None, valid: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Fond de chaque image : médiane des pixels valides de sa bordure.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, ny, nx)
        Images.
    width : int, optional
        Largeur de la bordure en pixels ; par défaut 10 % du plus petit côté,
        au moins 2.
    valid : ndarray of bool, shape (n_frames, ny, nx), optional
        Pixels valides ; les autres sont ignorés.

    Returns
    -------
    ndarray, shape (n_frames,)
        Fond de chaque image ; 0 si la bordure n'a aucun pixel valide.
    """
    n = width or max(2, min(frames.shape[1:]) // 10)
    border = np.zeros(frames.shape[1:], bool)
    border[:n], border[-n:], border[:, :n], border[:, -n:] = True, True, True, True
    out = np.empty(len(frames))
    for k, f in enumerate(frames):
        sel = border if valid is None else border & valid[k]
        out[k] = np.median(f[sel]) if sel.any() else 0.0
    return out


def save_image(path: PathLike, img: ArrayLike) -> None:
    """Écrit une image 2D ou 3D en float32, au format FITS ou TIFF selon l'extension.

    Parameters
    ----------
    path : str or Path
        Fichier de sortie ; ``.tif`` ou ``.tiff`` donne un TIFF, toute autre
        extension un FITS.
    img : array_like
        Image ou cube.
    """
    path = Path(path)
    data = np.asarray(img, dtype=np.float32)
    if path.suffix.lower() in TIFF_EXT:
        import tifffile

        tifffile.imwrite(path, data, photometric="minisblack")
    else:
        fits.writeto(path, data, overwrite=True)
