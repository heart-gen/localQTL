import os, sys, pathlib
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("READTHEDOCS", "True")

_MOCK_MODULES = [
    "torch", "cupy", "cudf", "rfmix_reader",
    "pyarrow", "pyarrow.parquet",
    "dask", "dask_cuda", "rapids_dask_dependency",
    "numba", "pandas_plink", "pgen",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, mock.MagicMock())

project = "localQTL"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "numpydoc",
]

autosummary_generate = True  # Generate stub pages from autosummary
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Mock heavy deps not available on RTD
autodoc_mock_imports = [
    "torch", "cupy", "cudf", "rfmix_reader",
    "pyarrow", "pyarrow.parquet",
    "dask", "dask_cuda", "rapids_dask_dependency",
    "numba", "pandas_plink", "pgen",
]

# Optional: keep type hints in descriptions for cleaner sigs
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
