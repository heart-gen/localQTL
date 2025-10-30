"""
This script was adapted from tensorQTL `core.py`:
https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
"""
import sys
import numpy as np
import pandas as pd

from .utils import gpu_available as _gpu_available_util

if _gpu_available_util():
    import cupy as cp
else:
    cp = np

sys.modules.setdefault("phenotypeio", sys.modules[__name__])

__all__ = [
    "read_phenotype_bed",
]


def gpu_available():
    """Return True if a GPU is available for phenotype IO operations."""
    cp_mod = globals().get("cp")
    if cp_mod is None or cp_mod is np:
        return False
    try:
        return cp_mod.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def read_phenotype_bed(path, as_tensor=False, device="cpu", dtype="float32"):
    """
    Load phenotype BED file into phenotype and position DataFrames or arrays.

    Parameters
    ----------
    path : str
        Path to BED-like file (.bed, .bed.gz, .bed.parquet, .parquet).
    as_tensor : bool, default=False
        If True, return phenotype values as NumPy/CuPy array instead of DataFrame.
    device : {"cpu","gpu"}
        Only used if as_tensor=True. Requires CuPy if device="gpu".
    dtype : str
        Data type for returned array.

    Returns
    -------
    phenotype : pd.DataFrame | np.ndarray | cp.ndarray
        Phenotype matrix (rows = phenotypes, cols = samples).
    pos_df : pd.DataFrame
        Position metadata:
        - 'chr' + 'pos' if start==end
        - otherwise 'chr','start','end'
    """
    # Load file
    if path.lower().endswith(('.bed.gz', '.bed')):
        df = pd.read_csv(path, sep="\t", dtype={"#chr": str, "#Chr": str})
    elif path.lower().endswith(('.bed.parquet', '.parquet')):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Normalize columns
    cols = [c.lower().replace("#chr", "chr").replace("#", "") for c in df.columns]
    df.columns = cols

    # Ensure phenotype ID is index
    if df.columns[3] != "id":
        df.rename(columns={df.columns[3]: "id"}, inplace=True)
    df.set_index("id", inplace=True)

    # Adjust coordinates
    if "start" in df.columns:
        df["start"] = df["start"].astype(int) + 1  # 1-based

    # Build position table
    pos_df = df[["chr", "start", "end"]].copy()
    df = df.drop(["chr", "start", "end"], axis=1)

    # Ensure consistent ordering of phenotypes and positions
    ordered_pos = pos_df.sort_values(["chr", "start", "end"])  # lexicographic
    if not pos_df.index.equals(ordered_pos.index):
        pos_df = ordered_pos
        df = df.loc[pos_df.index]

    # Collapse start==end to pos
    if (pos_df["start"] == pos_df["end"]).all():
        pos_df = pos_df[["chr", "end"]].rename(columns={"end": "pos"})

    # Convert to array/tensor if requested
    if as_tensor:
        arr = df.values.astype(dtype, copy=False)
        if device == "gpu":
            if cp is None or not gpu_available():
                raise RuntimeError("CuPy is installed but no GPU is available. Use device='cpu'")
            arr = cp.asarray(arr)
        return arr, pos_df
    else:
        return df, pos_df
