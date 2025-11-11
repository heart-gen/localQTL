import os, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

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
    "torch", "cupy", "cudf", "rfmix-reader", 
    "pyarrow", "dask_cuda", "numba", "pandas_plink",
    "dask"
]

# Optional: keep type hints in descriptions for cleaner sigs
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
