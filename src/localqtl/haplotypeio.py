"""
GPU-enabled utilities to incorporate local ancestry (RFMix) into tensorQTL-style
cis mapping. Provides:
  - RFMixReader: aligns RFMix local-ancestry to genotype variant order (lazy via dask)
  - InputGeneratorCisWithHaps: background-prefetched batch generator that yields
      phenotype, variants slice, haplotypes slice, their index ranges, and IDs

Notes
-----
- Designed for large-scale GPU eQTL with CuPy/cuDF where possible.
- Avoids materialization; uses dask-backed arrays and cuDF slicing.
- Compatible with original tensorQTL patterns while adding local ancestry.

Author: Kynon J Benjamin
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union

from rfmix_reader import read_rfmix

from .utils import gpu_available
from .genotypeio import InputGeneratorCis, background

if gpu_available():
    import cudf
    import cupy as cp
    from cudf import DataFrame as cuDF
    
    get_array_module = cp.get_array_module
else:
    cp = np
    cudf = pd
    cuDF = pd.DataFrame
    def get_array_module(x):
        return np

# ----------------------------
# Local ancestry readers
# ----------------------------
class RFMixReader:
    """Read and align RFMix local ancestry to variant grid.

    Parameters
    ----------
    prefix_path : str
        Directory containing RFMix per-chrom outputs and fb.tsv.
    select_samples : list[str], optional
        Subset of sample IDs to keep (order preserved).
    exclude_chrs : list[str], optional
        Chromosomes to exclude from imputed matrices.
    binary_path : str
        Path with prebuilt binary files (default: "./binary_files").
    verbose : bool
    dtype : numpy dtype

    Attributes
    ----------
    loci : cuDF
        Imputed loci aligned to variants (columns: ['chrom','pos','i','hap']).
    admix : dask.array
        Dask array with shape (loci, samples, ancestries)
    g_anc : cuDF or pd.DataFrame
        Sample metadata table from RFMix (contains 'sample_id', 'chrom').
    sample_ids : list[str]
    n_pops : int
    loci_df : pd.DataFrame
        Ancestry dosage aligned to hap_df.
    haplotypes : dask.array
        Haplotype-level ancestry matrix (variants x samples [x ancestries]).
    """

    def __init__(
        self, prefix_path: str, #variant_df: pd.DataFrame,
        select_samples: Optional[List[str]] = None,
        exclude_chrs: Optional[List[str]] = None,
        binary_path: str = "./binary_files",
        verbose: bool = True, dtype=np.int8
    ):
        # self.zarr_dir = f"{prefix_path}"
        bin_dir = f"{binary_path}"

        self.loci, self.g_anc, self.admix = read_rfmix(prefix_path,
                                                       binary_dir=bin_dir,
                                                       verbose=verbose)
        if self.admix.ndim != 3:
            n_vars, total = self.admix.shape
            n_pops = total // len(self.g_anc.sample_id.unique())
            n_samp = total // n_pops
            self.admix = self.admix.reshape(n_vars, n_samp, n_pops)

        # Guard unknown shapes
        if any(dim is None for dim in self.admix.shape):
            raise ValueError(
                "Ancestry array has unknown dimensions; expected (variants, samples, ancestries)."
            )

        # Build loci table
        self.loci = self.loci.rename(columns={"chromosome": "chrom",
                                              "physical_position": "pos"})
        self.loci["i"] = cudf.Series(range(len(self.loci)))
        self.loci["hap"] = self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str)

        # Subset samples
        self.sample_ids = _get_sample_ids(self.g_anc)
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.admix = self.admix[:, ix, :]
            if hasattr(self.g_anc, "to_arrow"):
                self.g_anc = self.g_anc.loc[ix].reset_index(drop=True)
            else:
                self.g_anc = self.g_anc.iloc[ix].reset_index(drop=True)
            self.sample_ids = _get_sample_ids(self.g_anc)

        # Exclude chromosomes if requested
        if exclude_chrs is not None and len(exclude_chrs) > 0:
            mask_pd = ~self.loci.to_pandas()["chrom"].isin(exclude_chrs).values
            self.admix = self.admix[mask_pd, :, :]
            keep_idx = np.nonzero(mask_pd)[0]
            self.loci = self.loci[keep_idx].reset_index(drop=True)
            self.loci["i"] = self.loci.index

        # Dimensions
        self.n_samples = int(self.admix.shape[1])
        self.n_pops = int(self.admix.shape[2])

        # Build hap tables
        if self.n_pops == 2:
            A0 = self.admix[:, :, [0]]
            loci_ids = (self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str) + "_A0")
            loci_df = self.loci.to_pandas()[["chrom", "pos"]].copy()
            loci_df["ancestry"] = 0
            loci_df["hap"] = _to_pandas(loci_ids)
            loci_df["index"] = np.arange(loci_df.shape[0])
            self.loci_df = loci_df.set_index("hap")
            self.loci_dfs = {c: g[["pos", "index"]].sort_values("pos").reset_index(drop=True)
                            for c, g in self.loci_df.reset_index().groupby("chrom", sort=False)}
            self.haplotypes = A0
        else: # >2 ancestries
            loci_dfs = []
            for anc in range(self.n_pops):
                loci_df_anc = self.loci.to_pandas()[["chrom", "pos"]].copy()
                loci_df_anc["ancestry"] = anc
                loci_df_anc["hap"] = (
                    loci_df_anc["chrom"].astype(str) + "_" + loci_df_anc["pos"].astype(str) + f"_A{anc}"
                )
                # Global index along flattened (variants*ancestries) axis
                loci_df_anc["index"] = np.arange(loci_df_anc.shape[0]) + anc * self.loci.shape[0]
                loci_dfs.append(loci_df_anc)

            self.loci_df = pd.concat(loci_dfs).set_index("hap")
            self.loci_dfs = {c: g[["pos", "index", "ancestry"]].sort_values("pos").reset_index(drop=True)
                            for c, g in self.loci_df.reset_index().groupby("chrom", sort=False)}
            self.haplotypes = self.admix  # dask array

    def load_haplotypes(self):
        """Force-load haplotype ancestry into memory as NumPy array."""
        return self.haplotypes.compute()

# -------------------------------
# Input generator for haplotypes
# -------------------------------
class InputGeneratorCisWithHaps(InputGeneratorCis):
    """
    Input generator for cis mapping (variants + local ancestry haplotypes).
    """

    def __init__(self, *args, haplotypes=None, loci_df=None,
                 on_the_fly_impute=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.haplotypes = haplotypes
        self.loci_df = loci_df.copy() if loci_df is not None else None
        if self.loci_df is not None:
            m = (self.variant_df.reset_index()
                 .merge(self.loci_df.reset_index(), on=["chrom","pos"], how="left"))
            if m["index"].isnull().any():
                raise ValueError("Some variants not found in loci_df for hap mapping.")
            self._geno2hap = m["index"].to_numpy(dtype=int)
        else:
            self._geno2hap = None
        self.on_the_fly_impute = on_the_fly_impute

    @staticmethod
    def _interpolate_block(block):
        """
        Interpolate missing values in a 3D haplotype block: (loci, samples, ancestries).

        Performs linear interpolation along the loci axis (axis=0) for each (sample, ancestry)
        pair independently. Supports NumPy or CuPy arrays via arr_mod.
        """
        # Determine arrray module
        mod = get_array_module(block)

        loci_dim, sample_dim, ancestry_dim = block.shape
        block = block.reshape(loci_dim, -1)  # Shape: (loci, samples * ancestries)
        idx = mod.arange(loci_dim)

        block_imputed = block.copy()

        for s in range(block.shape[1]):
            col = block[:, s]
            mask = mod.isnan(col)
            if mod.any(mask):
                valid = ~mask
                if mod.any(valid):
                    # Linear interpolation and rounding
                    interpolated = mod.interp(idx[mask], idx[valid], col[valid])
                    block_imputed[mask, s] = mod.round(interpolated).astype(int)

        return block_imputed.reshape(loci_dim, sample_dim, ancestry_dim)

    def _postprocess_batch(self, batch):
        if len(batch) == 4:
            p, G, v_idx, pid = batch
            if self._geno2hap is None:
                H_slice = self.haplotypes[v_idx, :, :]
            else:
                hap_idx = self._geno2hap[v_idx]
                H_slice = self.haplotypes[hap_idx, :, :]

            if self.on_the_fly_impute:
                H_block = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
                H = self._interpolate_block(H_block)
            else:
                H = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
            return p, G, v_idx, H, pid
        elif len(batch) == 5:
            p, G, v_idx, ids, group_id = batch
            if self._geno2hap is None:
                H_slice = self.haplotypes[v_idx, :, :]
            else:
                hap_idx = self._geno2hap[v_idx]
                H_slice = self.haplotypes[hap_idx, :, :]
  
            if self.on_the_fly_impute:
                H_block = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
                H = self._interpolate_block(H_block)
            else:
                H = H_slice.compute() if hasattr(H_slice, "compute") else np.asarray(H_slice)
            return p, G, v_idx, H, ids, group_id

    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False):
        for batch in super().generate_data(chrom=chrom, verbose=verbose):
            yield self._postprocess_batch(batch)


# ----------------------------
# Helpers functions
# ----------------------------
def _to_pandas(df: Union[cuDF, cudf.Series]) -> pd.DataFrame | pd.Series:
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _get_sample_ids(df: cuDF) -> List[str]:
    if hasattr(df, "to_arrow"):
        return df["sample_id"].to_arrow().to_pylist()
    return df["sample_id"].tolist()
