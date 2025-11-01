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
"""
import numpy as np
import pandas as pd
import torch
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

# Public exports
__all__ = [
    "RFMixReader",
    "InputGeneratorCisWithHaps",
]

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
        self.loci = _to_pandas(self.loci).rename(columns={"chromosome": "chrom",
                                                          "physical_position": "pos"})
        self.loci["i"] = np.arange(len(self.loci), dtype=int)
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
            loci_pd = _to_pandas(self.loci)
            mask_pd = ~loci_pd["chrom"].isin(exclude_chrs).values
            self.admix = self.admix[mask_pd, :, :]
            keep_idx = np.nonzero(mask_pd)[0]
            self.loci = loci_pd.iloc[keep_idx].reset_index(drop=True)
            self.loci["i"] = self.loci.index

        # Dimensions
        self.n_samples = int(self.admix.shape[1])
        self.n_pops = int(self.admix.shape[2])

        # Build hap tables
        if self.n_pops == 2:
            A0 = self.admix[:, :, [0]]
            loci_ids = (self.loci["chrom"].astype(str) + "_" + self.loci["pos"].astype(str) + "_A0")
            loci_df = _to_pandas(self.loci)[["chrom", "pos"]].copy()
            loci_df["ancestry"] = 0
            loci_df["hap"] = _to_pandas(loci_ids)
            loci_df["index"] = np.arange(loci_df.shape[0])
            self.loci_df = loci_df.set_index("hap")
            self.loci_dfs = {c: g[["pos", "index"]].sort_values("pos").reset_index(drop=True)
                            for c, g in self.loci_df.reset_index().groupby("chrom", sort=False)}
            self.haplotypes = A0
        else: # >2 ancestries
            loci_dfs = []
            loci_pd = _to_pandas(self.loci)
            for anc in range(self.n_pops):
                loci_df_anc = loci_pd[["chrom", "pos"]].copy()
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
                 on_the_fly_impute=True, preload_to_torch: bool = False,
                 torch_device: Union[str, torch.device, None] = None,
                 **kwargs):
        self.nearest_tolerance = kwargs.pop("nearest_tolerance", None) # in bp
        super().__init__(*args, **kwargs)

        if haplotypes is None:
            raise ValueError("`haplotypes` array is required for InputGeneratorCisWithHaps.")
        self.haplotypes = haplotypes
        try:
            n_loci = int(self.haplotypes.shape[0])
        except Exception as e:
            raise ValueError("`haplotypes` must have shape (loci, samples, ancestries).") from e

        self.on_the_fly_impute = on_the_fly_impute
        self._preload_to_torch = bool(preload_to_torch)
        self._hap_torch_cache: Optional[torch.Tensor] = None
        self._hap_torch_device: Optional[torch.device] = None
        self._hap_preloaded_is_gpu: bool = False
        try:
            self._hap_n_samples = int(self.haplotypes.shape[1])
            self._hap_n_ancestries = int(self.haplotypes.shape[2])
        except Exception:
            self._hap_n_samples, self._hap_n_ancestries = 0, 0

        if self._preload_to_torch:
            if torch_device is None:
                torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._hap_torch_device = torch.device(torch_device)
            base = haplotypes
            if hasattr(base, "compute"):
                base = base.compute()
            if isinstance(base, torch.Tensor):
                hap_tensor = base.to(device=self._hap_torch_device, dtype=torch.float32)
            else:
                if hasattr(base, "get"):
                    base = base.get()
                base_np = np.asarray(base, dtype=np.float32)
                hap_tensor = torch.as_tensor(base_np, dtype=torch.float32, device=self._hap_torch_device)
            self._hap_torch_cache = hap_tensor.contiguous()
            self._hap_preloaded_is_gpu = self._hap_torch_cache.is_cuda
            self._hap_n_samples = int(self._hap_torch_cache.shape[1])
            self._hap_n_ancestries = int(self._hap_torch_cache.shape[2])

        # Build mapping and mask
        self.loci_df = _to_pandas(loci_df).copy() if loci_df is not None else None
        if self.loci_df is not None:
            geno2hap, vmask, dropped_n = self.build_variant_to_locus_map(
                variant_df=self.variant_df, loci_df=self.loci_df,
                n_loci=n_loci, nearest_tolerance=self.nearest_tolerance,
            )
            self._geno2hap = geno2hap
            self._vmask = vmask
            self.n_unmapped_variants = int(dropped_n)

            if self.n_unmapped_variants:
                msg = (f"    ** dropping {self.n_unmapped_variants} "
                       "variants with no locus match")
                # Try to use parent's logger if present
                if hasattr(self, "logger") and getattr(self, "logger", None):
                    self.logger.info(msg)
                else:
                    print(msg)
        else:
            if n_loci != int(self.variant_df.shape[0]):
                raise ValueError(
                    f"haplotypes first dimension ({n_loci}) != variant_df rows "
                    f"({self.variant_df.shape[0]}). Provide `loci_df` to map."
                )
            self._geno2hap = np.arange(n_loci, dtype=int)
            self._vmask = np.ones(self.variant_df.shape[0], dtype=bool)
            self.n_unmapped_variants = 0
            
    @classmethod
    def build_variant_to_locus_map(
            cls, variant_df: pd.DataFrame, loci_df: pd.DataFrame,
            n_loci: int, nearest_tolerance: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Build a per-variant base locus index (0..n_loci-1) using per-chrom nearest join.
        """
        lb = loci_df.reset_index(drop=False)

        # Find an index-like column; normalize ancestry-flattened indices if present
        idx_col = next((c for c in ("index", "i", "locus_index") if c in lb.columns), None)
        if idx_col is None:
            raise ValueError("loci_df must include one of ['index','i','locus_index'].")

        idx = lb[idx_col].to_numpy()
        base_idx = (idx % n_loci).astype(int) if idx.max() >= n_loci else idx.astype(int)
        lb = lb.assign(index=base_idx)

        if not {"chrom", "pos"}.issubset(lb.columns):
            raise ValueError("loci_df must include 'chrom' and 'pos' columns.")

        loci_base = (lb[["chrom", "pos", "index"]]
                     .drop_duplicates(["chrom", "pos"], keep="first")
                     .reset_index(drop=True))

        if loci_base["index"].min() < 0 or loci_base["index"].max() >= n_loci:
            raise ValueError("Base 'index' out of bounds for haplotypes axis.")

        # Coerce chrom dtype and prep variants
        loci_base["chrom"] = loci_base["chrom"].astype(str)
        vreset = variant_df[["chrom", "pos"]].copy()
        vreset["chrom"] = vreset["chrom"].astype(str)
        vreset = (vreset.assign(variant_id=np.arange(len(vreset), dtype=int))
                         .sort_values(["chrom", "pos"])
                         .reset_index(drop=True))

        # Per-chrom nearest join (with optional tolerance)
        maps = []
        for c, vgrp in vreset.groupby("chrom", sort=False):
            lgrp = loci_base[loci_base["chrom"] == c].sort_values("pos")
            if lgrp.empty:
                tmp = vgrp.copy(); tmp["index"] = np.nan; maps.append(tmp); continue
            merged = pd.merge_asof(
                vgrp.sort_values("pos"), lgrp,
                on="pos", direction="nearest", allow_exact_matches=True,
                **({"tolerance": nearest_tolerance} if nearest_tolerance is not None else {})
            )
            maps.append(merged)

        m = (pd.concat(maps, axis=0)
               .sort_values("variant_id")
               .reset_index(drop=True))

        vmask     = ~m["index"].isna().to_numpy()
        dropped_n = int((~vmask).sum())

        geno2hap = np.full(len(m), -1, dtype=int)
        if vmask.any():
            geno2hap[vmask] = m.loc[vmask, "index"].to_numpy(dtype=int)
            
        return geno2hap, vmask, dropped_n

    @staticmethod
    def _interpolate_block(block):
        """
        Interpolate missing values in a 3D haplotype block: (loci, samples, ancestries).

        Performs linear interpolation along the loci axis (axis=0) for each (sample, ancestry)
        pair independently. Supports NumPy or CuPy arrays via arr_mod.
        """
        # Determine array module
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

        return block_imputed.reshape(loci_dim, sample_dim, ancestry_dim).astype(np.float32, copy=False)

    def _fetch_hap_block(self, v_idx: np.ndarray):
        # Map variants in this batch to locus indices
        if v_idx.size == 0:
            return self._empty_h_block()

        hap_idx = v_idx if self._geno2hap is None else self._geno2hap[v_idx]
        hap_idx = np.asarray(hap_idx, dtype=int)
        if self._preload_to_torch and self._hap_torch_cache is not None:
            index = torch.as_tensor(hap_idx, dtype=torch.long, device=self._hap_torch_cache.device)
            H_block = torch.index_select(self._hap_torch_cache, 0, index)
            if H_block.dtype != torch.float32:
                H_block = H_block.to(dtype=torch.float32)
            if self.on_the_fly_impute and torch.isnan(H_block).any():
                H_np = self._interpolate_block(H_block.detach().cpu().numpy())
                H_block = torch.as_tensor(H_np, dtype=torch.float32, device=self._hap_torch_cache.device).contiguous()
            if self.on_the_fly_impute and torch.isnan(H_block).any():
                raise ValueError("Detected NaNs in haplotype block after filtering.")
            return H_block

        H_slice = self.haplotypes[hap_idx, :, :]
        H_block = H_slice.compute() if hasattr(H_slice, "compute") else H_slice
        if hasattr(H_block, "get"):
            H_block = H_block.get()
        H_block = np.asarray(H_block, dtype=np.float32)
        if self.on_the_fly_impute and np.isnan(H_block).any():
            H_block = self._interpolate_block(H_block)
        if self.on_the_fly_impute and np.isnan(H_block).any():
            raise ValueError("Detected NaNs in haplotype block after filtering.")
        return H_block

    def _empty_h_block(self):
        """Return an empty H block with shape (0, samples, ancestries) and matching dtype."""
        n_samp = self._hap_n_samples if self._hap_n_samples is not None else 0
        n_anc = self._hap_n_ancestries if self._hap_n_ancestries is not None else 0
        if self._preload_to_torch and self._hap_torch_cache is not None:
            return torch.empty((0, n_samp, n_anc), dtype=self._hap_torch_cache.dtype,
                               device=self._hap_torch_cache.device)
        dtype = getattr(self.haplotypes, "dtype", np.float32)
        return np.empty((0, n_samp, n_anc), dtype=dtype)

    def _postprocess_batch(self, batch):
        """Preserve grouping contract from `InputGeneratorCis` and just append H."""
        if len(batch) == 4:
            p, G, v_idx, pid = batch
            vmask_local = self._vmask[v_idx]
            if not vmask_local.any():
                return p, G[vmask_local, :], v_idx[vmask_local], \
                    self._empty_h_block(), pid
            v_idx_f = v_idx[vmask_local]
            G_f = G[vmask_local, :]
            H = self._fetch_hap_block(v_idx_f)
            return p, G_f, v_idx_f, H, pid
        elif len(batch) == 5:
            p, G, v_idx, ids, group_id = batch
            vmask_local = self._vmask[v_idx]
            if not vmask_local.any():
                return p, G[vmask_local, :], v_idx[vmask_local], \
                    self._empty_h_block(), ids, group_id
            v_idx_f = v_idx[vmask_local]
            G_f = G[vmask_local, :]
            H = self._fetch_hap_block(v_idx_f)
            return p, G_f, v_idx_f, H, ids, group_id
        else:
            raise ValueError(f"Unexpected batch structure from base generator: len={len(batch)}")


# ----------------------------
# Helpers functions
# ----------------------------
def _to_pandas(df: Union[cuDF, cudf.Series]) -> pd.DataFrame | pd.Series:
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _get_sample_ids(df: cuDF) -> List[str]:
    if hasattr(df, "to_arrow"):
        return df["sample_id"].to_arrow().to_pylist()
    return df["sample_id"].tolist()
