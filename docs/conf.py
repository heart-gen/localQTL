import os, sys
from datetime import datetime

project = "localQTL"
copyright = f"{datetime.now():%Y}, localQTL"
author = "localQTL contributors"

sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"

source_suffix = {
    ".rst": "restructuredtext",
}

master_doc = "index"
