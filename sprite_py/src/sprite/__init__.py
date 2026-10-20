"""SPRITE : *SParse Recovery of InsTrumental rEsponse* (Ngolè Mboula et al. 2015).

Portage Python du code C++ SPRITE v1.0 du CEA, construit sur ModOpt.

Point d'entrée principal : :func:`sprite`, paramétré par :class:`SpriteConfig`.
"""

from .solver import SpriteConfig, SpriteResult, sprite

__all__ = ["SpriteConfig", "SpriteResult", "sprite"]
__version__ = "1.0.0"
