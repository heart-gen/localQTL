from __future__ import annotations
import os, sys, pathlib, types
from datetime import datetime

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("READTHEDOCS", "True")

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib 

PYPROJECT: dict = {}
try:
    with open(ROOT_DIR / "pyproject.toml", "rb") as f:
        PYPROJECT = tomllib.load(f)
except Exception:
    pass

meta = PYPROJECT.get("project") or PYPROJECT.get("tool", {}).get("poetry") or {}
project = meta.get("name", "localQTL")
authors = meta.get("authors", [])
if isinstance(authors, list) and authors and isinstance(authors[0], dict):
    author = ", ".join(a.get("name", "") for a in authors if isinstance(a, dict)) or "localQTL Developers"
else:
    author = meta.get("authors", "localQTL Developers")

current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

# Read version
def _read_version() -> str:
    from importlib.metadata import version as _v, PackageNotFoundError
    candidates = {project, project.replace("-", "_"), project.replace("_", "-")}
    # Also try lowercase variants (some tools normalize)
    candidates |= {c.lower() for c in list(candidates)}
    for dist in candidates:
        try:
            return _v(dist)
        except PackageNotFoundError:
            continue
    return meta.get("version", "0.0.0")

release = _read_version()
version = release

# Mock heavy deps not available on RTD
STUB_MODULES = [
    "torch", "cupy", "cudf", "rfmix_reader",
    "pyarrow", "pyarrow.parquet",
    "dask", "dask_cuda", "rapids_dask_dependency",
    "numba", "pandas_plink", "pgen",
]
for name in STUB_MODULES:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "numpydoc",
    "sphinx_copybutton",
]

# RST docs
source_suffix = [".rst"]
root_doc = "index"

# Autodoc / Autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
python_use_unqualified_type_names = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# Avoid duplicate member sections when using numpydoc + napoleon
numpydoc_show_class_members = False

# Mock heavy/optional deps when autodoc imports modules
autodoc_mock_imports = [
    "torch", "cupy", "cudf", "rfmix_reader",
    "pyarrow", "pyarrow.parquet",
    "dask", "dask_cuda", "rapids_dask_dependency",
    "numba", "pandas_plink", "pgen",
]

# Intersphinx cross-links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# HTML
html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}
html_title = f"{project} {release}"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Ensure _static exists to avoid warnings on fresh repos
_static = pathlib.Path(__file__).resolve().parent / "_static"
_static.mkdir(exist_ok=True)
html_static_path = ["_static"]

# Quality gate
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "pandas.Series"),
]
