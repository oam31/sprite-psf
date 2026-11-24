# Configuration de la documentation Sphinx de sprite-psf.
#
# Sources : les documents techniques en Markdown (MyST) dans technique/,
# le CHANGELOG de la racine (inclus par changelog.md) et la référence de
# l'API, générée par sphinx-autoapi dans api/ (non versionnée).
# Construction locale : `make html` dans ce répertoire → _build/html.

import re
from pathlib import Path

# Version lue dans le paquet sans l'importer (l'import exige ModOpt ;
# autoapi analyse le code sans l'exécuter).
_init = Path(__file__).resolve().parents[1] / "sprite_py" / "src" / "sprite" / "__init__.py"
_version = re.search(r'__version__ = "([^"]+)"', _init.read_text()).group(1)

project = "sprite-psf"
copyright = "2026, Olivier Amram"
author = "Olivier Amram"
release = _version
version = _version

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "document_technique.md"]

language = "fr"

# -- MyST ----------------------------------------------------------------------
# Ancres des titres identiques à celles de GitHub : les liens internes des
# documents fonctionnent dans les deux rendus.
myst_heading_anchors = 5
myst_enable_extensions = ["dollarmath", "amsmath", "deflist"]
# Blocs ```math (syntaxe des formules de GitHub) rendus comme des formules.
myst_fence_as_directive = ["math"]

# -- Sortie HTML ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- API (sphinx-autoapi) ------------------------------------------------------
autoapi_dirs = ["../sprite_py/src"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_keep_files = False
autodoc_typehints = "signature"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
