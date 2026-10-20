"""Exécution parallèle de reconstructions indépendantes (contrôles, balayages).

Les contrôles de :mod:`sprite.diagnostics` enchaînent des reconstructions
indépendantes : blocs de la prédiction croisée, moitiés, valeurs d'un balayage.
Elles sont réparties sur des processus séparés (le calcul est en Python et
numpy ; des threads se gêneraient). Chaque processus est limité à un thread de
calcul numérique pour que les processus ne se disputent pas les cœurs.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Sequence

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def default_jobs() -> int:
    """Nombre de processus par défaut : cœurs disponibles moins un (au moins 1)."""
    return max(1, (os.cpu_count() or 2) - 1)


def run_parallel(func: Callable[..., Any], tasks: Sequence[tuple], n_jobs: int | None = None) -> list[Any]:
    """Applique ``func`` à chaque tuple d'arguments de ``tasks``, en parallèle si possible.

    Parameters
    ----------
    func : callable
        Fonction de niveau module (sérialisable par ``pickle``).
    tasks : sequence of tuple
        Arguments de chaque appel.
    n_jobs : int, optional
        Nombre de processus ; :func:`default_jobs` par défaut, borné par le
        nombre de tâches. ``1`` exécute tout dans le processus courant.

    Returns
    -------
    list
        Résultats, dans l'ordre de ``tasks``.
    """
    n_jobs = min(default_jobs() if n_jobs is None else n_jobs, len(tasks))
    if n_jobs <= 1:
        return [func(*args) for args in tasks]
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: "1" for v in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(func, *args) for args in tasks]
            return [f.result() for f in futures]
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
