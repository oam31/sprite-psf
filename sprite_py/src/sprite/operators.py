"""Opérateur d'observation ``M`` de SPRITE et son adjoint.

Pour chaque image basse résolution (BR) ``k``, le modèle direct du papier
(éq. 14) est ``y_k = f_k D H_k x`` : ``H_k`` est une convolution circulaire par
un noyau de Lanczos qui code le décalage sous-pixel de l'image ``k``, ``D``
garde un pixel haute résolution (HR) sur ``upfact`` dans chaque direction et
``f_k`` est le flux photométrique. Chaque image est blanchie par son niveau de
bruit ``sigma_k``, comme dans le C++.

Tous les tableaux suivent l'orientation naturelle de numpy : ``(ligne,
colonne) = (y, x)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lanczos_1d(u: float, radius: int) -> NDArray[np.float64]:
    """Coefficients d'un noyau de Lanczos 1D décalé.

    Parameters
    ----------
    u : float
        Décalage, en pixels.
    radius : int
        Rayon du noyau ``a`` ; le noyau a ``2 a + 1`` coefficients.

    Returns
    -------
    ndarray, shape (2 * radius + 1,)
        ``L(u - t)`` pour ``t = -radius .. radius``, avec
        ``L(x) = sinc(x) sinc(x / a)``.
    """
    t = u - np.arange(-radius, radius + 1)
    return np.sinc(t) * np.sinc(t / radius)


def lanczos_kernel(shift: ArrayLike, radius: int = 10) -> NDArray[np.float64]:
    """Noyau de Lanczos 2D séparable.

    Parameters
    ----------
    shift : array_like, shape (2,)
        Décalage ``(dy, dx)`` en pixels HR.
    radius : int, default 10
        Rayon du noyau (valeur du C++).

    Returns
    -------
    ndarray, shape (2 * radius + 1, 2 * radius + 1)
        Noyau non normalisé.
    """
    dy, dx = np.asarray(shift, dtype=float)
    return np.outer(lanczos_1d(dy, radius), lanczos_1d(dx, radius))


def kernel_otf(kernel: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.complex128]:
    """Fonction de transfert d'un noyau normalisé et centré sur ``shape // 2``.

    Reproduit iSAP ``psf_convol(..., FluxNorm=True, PsfMaxShift=False)`` : le
    noyau est placé au centre d'une image de taille ``shape`` et normalisé à
    une somme unité.

    Parameters
    ----------
    kernel : ndarray, shape (ky, kx)
        Noyau de convolution.
    shape : tuple of int
        Taille de l'image à convoluer.

    Returns
    -------
    ndarray, shape (shape[0], shape[1] // 2 + 1)
        Transformée de Fourier réelle du noyau centré à l'origine.
    """
    pad = np.zeros(shape)
    ny, nx = kernel.shape
    oy = int(np.ceil((shape[0] - ny) / 2))
    ox = int(np.ceil((shape[1] - nx) / 2))
    pad[oy:oy + ny, ox:ox + nx] = kernel
    pad /= pad.sum()
    return np.fft.rfft2(np.fft.ifftshift(pad))


def convolve(img: NDArray[np.float64], otf: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convolution circulaire par une fonction de transfert de :func:`kernel_otf`.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image.
    otf : ndarray
        Fonction de transfert calculée pour ``img.shape``.

    Returns
    -------
    ndarray, shape (ny, nx)
        Image convoluée.
    """
    return np.fft.irfft2(np.fft.rfft2(img) * otf, s=img.shape)


def decimate(img: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Garde un pixel sur ``factor`` dans chaque direction (opérateur ``D``).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image HR.
    factor : int
        Facteur de décimation.

    Returns
    -------
    ndarray, shape (ny // factor, nx // factor)
        Pixels d'indices ``(factor i, factor j)``.
    """
    return img[::factor, ::factor]


def zero_insert(img: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Adjoint de :func:`decimate` (iSAP ``transpose_decim``).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image BR.
    factor : int
        Facteur de sur-échantillonnage.

    Returns
    -------
    ndarray, shape (ny * factor, nx * factor)
        Image HR nulle sauf aux pixels ``(factor i, factor j)``.
    """
    out = np.zeros((img.shape[0] * factor, img.shape[1] * factor))
    out[::factor, ::factor] = img
    return out


class SpriteOperator:
    """Opérateur direct blanchi ``M`` et son adjoint.

    Parameters
    ----------
    shifts : array_like, shape (n_frames, 2)
        Décalages ``(dy, dx)`` de chaque image, en pixels HR.
    flux : array_like, shape (n_frames,)
        Flux photométriques relatifs.
    sigma : array_like, shape (n_frames,)
        Écart type du bruit de chaque image BR (poids de blanchiment).
    lr_shape : tuple of int
        Taille d'une image BR.
    upfact : int, default 2
        Facteur de sur-échantillonnage.
    lanczos_radius : int, default 10
        Rayon des noyaux de recalage (valeur du C++).
    """

    def __init__(
        self,
        shifts: ArrayLike,
        flux: ArrayLike,
        sigma: ArrayLike,
        lr_shape: tuple[int, int],
        upfact: int = 2,
        lanczos_radius: int = 10,
    ) -> None:
        self.upfact = upfact
        self.flux = np.asarray(flux, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.lr_shape = tuple(lr_shape)
        self.hr_shape = (lr_shape[0] * upfact, lr_shape[1] * upfact)
        self.kernels = np.array([lanczos_kernel(s, lanczos_radius) for s in np.asarray(shifts)])
        self._otf = np.array([kernel_otf(k, self.hr_shape) for k in self.kernels])
        # Adjoint : convolution par le noyau tourné de 180 degrés.
        self._otf_adj = np.array([kernel_otf(k[::-1, ::-1], self.hr_shape) for k in self.kernels])
        self._scale = self.flux / self.sigma

    @property
    def n_frames(self) -> int:
        """Nombre d'images BR."""
        return self.flux.size

    def op(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applique ``M``.

        Parameters
        ----------
        x : ndarray, shape hr_shape
            Image HR.

        Returns
        -------
        ndarray, shape (n_frames,) + lr_shape
            Images BR prédites, blanchies (divisées par ``sigma``).
        """
        fx = np.fft.rfft2(x)
        out = np.empty((self.n_frames,) + self.lr_shape)
        for k in range(self.n_frames):
            hk = np.fft.irfft2(fx * self._otf[k], s=self.hr_shape)
            out[k] = decimate(hk, self.upfact) * self._scale[k]
        return out

    def adj_op(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applique l'adjoint ``M^T``.

        Parameters
        ----------
        y : ndarray, shape (n_frames,) + lr_shape
            Images BR blanchies.

        Returns
        -------
        ndarray, shape hr_shape
            Image HR.
        """
        acc = np.zeros((self.hr_shape[0], self.hr_shape[1] // 2 + 1), dtype=complex)
        for k in range(self.n_frames):
            up = zero_insert(y[k] / self.sigma[k], self.upfact)
            acc += np.fft.rfft2(up) * self._otf_adj[k] * self.flux[k]
        return np.fft.irfft2(acc, s=self.hr_shape)

    def normal_op(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applique ``M^T M`` (``sprite_global_op_mat``).

        Parameters
        ----------
        x : ndarray, shape hr_shape
            Image HR.

        Returns
        -------
        ndarray, shape hr_shape
            ``M^T M x``.
        """
        return self.adj_op(self.op(x))

    def spectral_radius(
        self, tol: float = 1e-3, max_iter: int = 100, rng: np.random.Generator | None = None
    ) -> tuple[float, int]:
        """Plus grande valeur propre de ``M^T M`` par la méthode de la puissance.

        Reproduit ``Sprite::spectral_rad_est``, y compris la marge finale
        ``(1 + tol)``. Le C++ part de valeurs ``rand()`` ; ici le vecteur initial
        vient de ``rng`` (graine fixée par défaut).

        Parameters
        ----------
        tol : float, default 1e-3
            Tolérance relative d'arrêt.
        max_iter : int, default 100
            Nombre maximal d'itérations.
        rng : numpy.random.Generator, optional
            Générateur du vecteur initial ; graine 0 par défaut.

        Returns
        -------
        rho : float
            Rayon spectral estimé, majoré de ``(1 + tol)``.
        n_iter : int
            Nombre d'itérations effectuées.
        """
        rng = np.random.default_rng(0) if rng is None else rng
        x = rng.random(self.hr_shape)
        x /= np.linalg.norm(x)
        lam_old, lam = 0.0, 1.0
        n_iter = 0
        while abs(lam_old - lam) / lam > tol and n_iter < max_iter:
            x = self.normal_op(x / lam)
            lam_old, lam = lam, float(np.linalg.norm(x))
            n_iter += 1
        return lam * (1 + tol), n_iter
