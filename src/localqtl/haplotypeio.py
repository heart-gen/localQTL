from __future__ import annotations
"""
Local ancestry (RFMix, FLARE, haptools) for tensorQTL-style cis mapping.

Provides:
  - LocalAncestryPanel: an rfmix_reader Dataset served one chromosome at a
      time -- per-chromosome int8 count arrays plus a segment index per SNP
  - RFMixReader / FlareReader / open_panel: build a panel from RFMix output,
      FLARE output or an existing rfmix_reader Zarr cache
  - InputGeneratorCisWithHaps: background-prefetched batch generator that
      yields phenotype, variant slice, haplotype slice, index ranges and IDs

Notes
-----
- Local ancestry is piecewise constant, so a genotype variant maps to the
  RFMix segment (``.msp.tsv``) or locus that covers it; the panel holds only
  the current chromosome in memory (chr1 for 500 samples is ~1 GB from
  ``.fb.tsv`` loci, a few MB from ``.msp.tsv`` segments) and per-gene windows
  are numpy / torch slices of it.
- A plain ``(loci, samples, ancestries)`` array plus ``loci_df`` is still
  accepted (legacy path, nearest-position join).
"""
import warnings
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Sequence, Union

from .utils import select_array_module
from .genotypeio import InputGeneratorCis, background

def _get_array_module(x):
    try:
        backend = select_array_module()
        if hasattr(backend, "get_array_module"):
            return backend.get_array_module(x)
    except Exception:
        pass
    return np


# Public exports
__all__ = [
    "LocalAncestryPanel",
    "RFMixReader",
    "FlareReader",
    "open_panel",
    "InputGeneratorCisWithHaps",
]

# ----------------------------
# Local ancestry panel
# ----------------------------
class LocalAncestryPanel:
    """
    Local ancestry as an ``rfmix_reader`` Dataset, served one chromosome at a
    time.

    Parameters
    ----------
    ds : xarray.Dataset
        From ``rfmix_reader.open_rfmix`` / ``open_flare`` / ``open_simu`` /
        ``open_local_ancestry``.
    select_samples : list[str], optional
        Subset (and order) of sample IDs to keep; use the genotype columns.
    exclude_chrs : list[str], optional
        Chromosomes to drop.
    collapse_two_ancestries : bool, default True
        With exactly two ancestries keep only the first channel (the second
        is ``2 - first``, so both would be collinear with the intercept).

    Attributes
    ----------
    ds : xarray.Dataset
    sample_ids : list[str]
    ancestries : list[str]
        Ancestry labels in the tool's order.
    channels : list[int]
        Ancestry indices exposed as haplotype channels.
    n_samples, n_pops, n_channels : int
    g_anc : pd.DataFrame
        Global ancestry per sample (and chromosome).
    shape : tuple
        ``(n_variants, n_samples, n_channels)``.
    """

    def __init__(self, ds, select_samples: Optional[Sequence[str]] = None,
                 exclude_chrs: Optional[Sequence[str]] = None,
                 collapse_two_ancestries: bool = True):
        import rfmix_reader.core  # noqa: F401  (registers ds.la)
        from rfmix_reader.formats.common import normalize_chrom_label

        if select_samples is not None:
            ids = ds.la.samples
            missing = [s for s in select_samples if s not in ids]
            if missing:
                raise ValueError(f"{len(missing)} selected samples not in the local ancestry "
                                 f"panel (first: {missing[:5]})")
            lookup = {s: i for i, s in enumerate(ids)}
            ds = ds.isel(sample=[lookup[s] for s in select_samples])
        if exclude_chrs:
            drop = {normalize_chrom_label(str(c)) for c in exclude_chrs}
            labels = np.array([normalize_chrom_label(str(c)) for c in ds.chromosome.values])
            ds = ds.isel(variant=np.flatnonzero(~np.isin(labels, list(drop))))

        self.ds = ds
        self.sample_ids: List[str] = ds.la.samples
        self.ancestries: List[str] = ds.la.ancestries
        self.n_samples = int(ds.la.n_samples)
        self.n_pops = len(self.ancestries)
        self.channels = [0] if (self.n_pops == 2 and collapse_two_ancestries) else list(range(self.n_pops))
        self.n_channels = len(self.channels)
        self.chromosomes: List[str] = ds.la.chromosomes
        self.g_anc = ds.la.global_ancestry
        self._cache_chrom: Optional[str] = None
        self._cache: Optional[np.ndarray] = None
        self._loci: Optional[pd.DataFrame] = None
        self._loci_df: Optional[pd.DataFrame] = None

    # -- shape / legacy compatibility -------------------------------------
    @property
    def shape(self):
        return (int(self.ds.la.n_variants), self.n_samples, self.n_channels)

    @property
    def haplotypes(self):
        """The panel itself (``haplotypes.shape[2]`` gives the channel count)."""
        return self

    @property
    def loci(self) -> pd.DataFrame:
        """Per-variant table: ``chrom, pos, end, i, hap`` (``end`` = segment end)."""
        if self._loci is None:
            ds = self.ds
            chrom = np.asarray(ds.chromosome.values).astype(str)
            pos = np.asarray(ds.variant_position.values, dtype=np.int64)
            end = (np.asarray(ds.segment_end.values, dtype=np.int64)
                   if "segment_end" in ds.coords else pos)
            df = pd.DataFrame({"chrom": chrom, "pos": pos, "end": end})
            df["i"] = np.arange(len(df), dtype=int)
            df["hap"] = df["chrom"] + "_" + df["pos"].astype(str)
            self._loci = df
        return self._loci

    @property
    def loci_df(self) -> pd.DataFrame:
        """Legacy per-(locus, channel) table indexed by ``hap`` with ``index`` into
        the flattened ``(variants * channels)`` axis (as ``load_haplotypes()``)."""
        if self._loci_df is None:
            base = self.loci[["chrom", "pos"]]
            n = len(base)
            parts = []
            for k, anc in enumerate(self.channels):
                d = base.copy()
                d["ancestry"] = anc
                d["hap"] = d["chrom"] + "_" + d["pos"].astype(str) + f"_A{anc}"
                d["index"] = np.arange(n) + k * n
                parts.append(d)
            self._loci_df = pd.concat(parts).set_index("hap")
        return self._loci_df

    # -- per-chromosome access ---------------------------------------------
    def chrom_counts(self, chrom: str) -> np.ndarray:
        """
        ``(variants_on_chrom, samples, channels)`` int8 diploid ancestry counts
        (``-1`` = no call), materialised once and cached until another
        chromosome is requested.
        """
        key = str(chrom)
        if self._cache_chrom != key:
            sub = self.ds.la.sel_chrom(key)
            counts = np.asarray(sub.la.counts.values)
            if self.channels != list(range(self.n_pops)):
                counts = np.ascontiguousarray(counts[:, :, self.channels])
            self._cache_chrom, self._cache = key, counts
        return self._cache

    def locus_index(self, chrom: str, positions, method: str = "stepwise",
                    tolerance: Optional[int] = None) -> np.ndarray:
        """Row of :meth:`chrom_counts` covering each position (``-1`` = none)."""
        try:
            sub = self.ds.la.sel_chrom(str(chrom))
        except KeyError:
            return np.full(np.asarray(positions).size, -1, dtype=np.int64)
        return sub.la.locus_index(str(chrom), positions, method=method, tolerance=tolerance)

    def load_haplotypes(self) -> np.ndarray:
        """
        Whole-genome ``(variants, samples, channels)`` int8 array (``-1`` = no
        call).  Convenient for small data; per-chromosome access through the
        generator is the intended path for large cohorts.
        """
        counts = np.asarray(self.ds.la.counts.values)
        if self.channels != list(range(self.n_pops)):
            counts = np.ascontiguousarray(counts[:, :, self.channels])
        return counts


class RFMixReader(LocalAncestryPanel):
    """
    Local ancestry from RFMix output (``.msp.tsv`` segments by default).

    Parameters
    ----------
    prefix_path : str
        Directory, file or path prefix of the RFMix output.
    select_samples, exclude_chrs, collapse_two_ancestries
        See :class:`LocalAncestryPanel`.
    source : {"msp", "fb"}
        ``msp`` (segment hard calls, recommended for QTL mapping) or ``fb``.
    cache_dir : str, optional
        rfmix_reader Zarr cache (parsed once, reopened instantly afterwards).
    chrom : str, optional
        Restrict to one chromosome.
    verbose : bool
    """

    def __init__(self, prefix_path: str, select_samples: Optional[Sequence[str]] = None,
                 exclude_chrs: Optional[Sequence[str]] = None, *, source: str = "msp",
                 cache_dir: Optional[str] = None, chrom: Optional[str] = None,
                 collapse_two_ancestries: bool = True, verbose: bool = True,
                 binary_path: Optional[str] = None, **open_kwargs):
        from rfmix_reader import open_rfmix

        if binary_path is not None:
            warnings.warn("binary_path is ignored: rfmix-reader >= 0.6 uses a Zarr cache "
                          "(cache_dir=...) instead of .bin files.", DeprecationWarning, stacklevel=2)
        ds = open_rfmix(prefix_path, source=source, chrom=chrom, cache_dir=cache_dir,
                        verbose=verbose, **open_kwargs)
        super().__init__(ds, select_samples=select_samples, exclude_chrs=exclude_chrs,
                         collapse_two_ancestries=collapse_two_ancestries)


class FlareReader(LocalAncestryPanel):
    """Local ancestry from FLARE output (``.anc.vcf.gz`` + ``.global.anc.gz``)."""

    def __init__(self, prefix_path: str, select_samples: Optional[Sequence[str]] = None,
                 exclude_chrs: Optional[Sequence[str]] = None, *,
                 cache_dir: Optional[str] = None, chrom: Optional[str] = None,
                 collapse_two_ancestries: bool = True, verbose: bool = True, **open_kwargs):
        from rfmix_reader import open_flare

        open_kwargs.pop("chunk_size", None)      # legacy argument
        ds = open_flare(prefix_path, chrom=chrom, cache_dir=cache_dir, verbose=verbose, **open_kwargs)
        super().__init__(ds, select_samples=select_samples, exclude_chrs=exclude_chrs,
                         collapse_two_ancestries=collapse_two_ancestries)


def open_panel(cache_dir: str, select_samples: Optional[Sequence[str]] = None,
               exclude_chrs: Optional[Sequence[str]] = None, *, chrom: Optional[str] = None,
               collapse_two_ancestries: bool = True) -> LocalAncestryPanel:
    """Panel from an existing rfmix_reader Zarr cache (``rfmix-reader convert ...``)."""
    from rfmix_reader import open_local_ancestry

    ds = open_local_ancestry(cache_dir, chrom=chrom)
    return LocalAncestryPanel(ds, select_samples=select_samples, exclude_chrs=exclude_chrs,
                              collapse_two_ancestries=collapse_two_ancestries)


def _is_panel(obj) -> bool:
    """Duck-typed :class:`LocalAncestryPanel` check (the package may be imported under two names)."""
    return all(hasattr(obj, a) for a in ("chrom_counts", "locus_index", "sample_ids", "n_channels"))


def _fill_missing_along_loci(counts: np.ndarray) -> np.ndarray:
    """
    Chromosome-wide imputation of missing calls (``-1``) by rounded linear
    interpolation along the loci axis, per (sample, channel) column; columns
    without any call stay ``-1``.  Same rule as ``_interpolate_block`` but
    applied once per chromosome, so windows never see a NaN that has a
    neighbour anywhere on the chromosome.
    """
    missing = counts < 0
    if not missing.any():
        return counts
    out = counts.copy()
    n = counts.shape[0]
    flat = out.reshape(n, -1)
    miss = missing.reshape(n, -1)
    idx = np.arange(n)
    for j in np.flatnonzero(miss.any(axis=0)):
        m = miss[:, j]
        valid = ~m
        if valid.any():
            flat[m, j] = np.rint(np.interp(idx[m], idx[valid], flat[valid, j].astype(float))).astype(counts.dtype)
    return out


# ----------------------------
# Input generator
# ----------------------------
class InputGeneratorCisWithHaps(InputGeneratorCis):
    """
    Input generator for cis mapping (variants + local ancestry haplotypes).

    ``haplotypes`` is either a :class:`LocalAncestryPanel` (recommended: one
    chromosome of counts in memory at a time, exact segment lookup per SNP)
    or a ``(loci, samples, channels)`` array with a ``loci_df`` (legacy:
    nearest-position join, whole array in memory).

    Parameters
    ----------
    locus_method : {"stepwise", "nearest"}
        Panel mode: how a SNP is matched to a local-ancestry locus
        (``stepwise`` = the segment covering it; ``nearest`` = closest locus,
        within ``nearest_tolerance`` bp if given).
    nearest_tolerance : int, optional
        Maximum distance for ``nearest`` matches (legacy path: merge tolerance).
    on_the_fly_impute : bool
        Fill missing calls in a window by interpolation along the loci axis.
    preload_to_torch : bool
        Keep the (per-chromosome) haplotype array as a torch tensor on
        ``torch_device`` and slice windows there.
    """

    def __init__(self, *args, haplotypes=None, loci_df=None,
                 on_the_fly_impute=True, preload_to_torch: bool = False,
                 torch_device: Union[str, torch.device, None] = None,
                 locus_method: str = "stepwise", **kwargs):
        self.nearest_tolerance = kwargs.pop("nearest_tolerance", None) # in bp
        super().__init__(*args, **kwargs)

        if haplotypes is None:
            raise ValueError("`haplotypes` array or LocalAncestryPanel is required for InputGeneratorCisWithHaps.")
        self.haplotypes = haplotypes
        self.on_the_fly_impute = on_the_fly_impute
        self._preload_to_torch = bool(preload_to_torch)
        self._hap_torch_cache: Optional[torch.Tensor] = None
        self._hap_torch_device: Optional[torch.device] = None
        self._hap_preloaded_is_gpu: bool = False
        if self._preload_to_torch:
            if torch_device is None:
                torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._hap_torch_device = torch.device(torch_device)

        self._panel: Optional[LocalAncestryPanel] = haplotypes if _is_panel(haplotypes) else None
        if self._panel is not None:
            self._init_panel_mode(locus_method)
        else:
            self._init_array_mode(haplotypes, loci_df)

    # -- panel mode --------------------------------------------------------
    def _init_panel_mode(self, locus_method: str):
        panel = self._panel
        geno_samples = [str(s) for s in self.genotype_df.columns]
        if panel.sample_ids != geno_samples:
            raise ValueError(
                "Local ancestry samples do not match the genotype columns (order included); "
                "build the panel with select_samples=list(genotype_df.columns).")
        self._hap_n_samples = panel.n_samples
        self._hap_n_ancestries = panel.n_channels
        if self.nearest_tolerance is not None and locus_method == "stepwise":
            locus_method = "nearest"
        self.locus_method = locus_method

        vchrom = self.variant_df["chrom"].astype(str).to_numpy()
        vpos = self.variant_df["pos"].to_numpy(dtype=np.int64)
        geno2hap = np.full(len(vchrom), -1, dtype=np.int64)
        for c in pd.unique(vchrom):
            m = np.flatnonzero(vchrom == c)
            geno2hap[m] = panel.locus_index(c, vpos[m], method=locus_method,
                                            tolerance=self.nearest_tolerance)
        self._var_chrom = vchrom
        self._geno2hap = geno2hap
        self._vmask = geno2hap >= 0
        self.n_unmapped_variants = int((~self._vmask).sum())
        if self.n_unmapped_variants:
            msg = f"    ** dropping {self.n_unmapped_variants} variants with no local-ancestry locus"
            if getattr(self, "logger", None):
                self.logger.info(msg)
            else:
                print(msg)
        self._chrom_loaded: Optional[str] = None
        self._chrom_np: Optional[np.ndarray] = None

    def _ensure_chrom(self, chrom: str):
        """Load ``chrom``'s counts (numpy int8, and torch if preloading) once."""
        if self._chrom_loaded == chrom:
            return
        counts = self._panel.chrom_counts(chrom)
        if self.on_the_fly_impute:
            counts = _fill_missing_along_loci(counts)
        self._chrom_np = counts
        if self._preload_to_torch:
            self._hap_torch_cache = torch.as_tensor(counts, device=self._hap_torch_device)
            self._hap_preloaded_is_gpu = self._hap_torch_cache.is_cuda
        self._chrom_loaded = chrom

    def _fetch_hap_block_panel(self, v_idx: np.ndarray):
        chrom = str(self._var_chrom[v_idx[0]])
        self._ensure_chrom(chrom)
        hap_idx = self._geno2hap[v_idx]
        if self._preload_to_torch:
            index = torch.as_tensor(hap_idx, dtype=torch.long, device=self._hap_torch_cache.device)
            H_block = torch.index_select(self._hap_torch_cache, 0, index).to(dtype=torch.float32)
            H_block = torch.where(H_block < 0, torch.nan, H_block)
            if self.on_the_fly_impute and torch.isnan(H_block).any():
                H_np = self._interpolate_block(H_block.detach().cpu().numpy())
                H_block = torch.as_tensor(H_np, dtype=torch.float32, device=self._hap_torch_cache.device).contiguous()
            if self.on_the_fly_impute and torch.isnan(H_block).any():
                raise ValueError("Detected NaNs in haplotype block after filtering.")
            return H_block
        H_block = self._chrom_np[hap_idx].astype(np.float32)
        H_block[H_block < 0] = np.nan
        if self.on_the_fly_impute and np.isnan(H_block).any():
            H_block = self._interpolate_block(H_block)
        if self.on_the_fly_impute and np.isnan(H_block).any():
            raise ValueError("Detected NaNs in haplotype block after filtering.")
        return H_block

    # -- legacy array mode -------------------------------------------------
    def _init_array_mode(self, haplotypes, loci_df):
        try:
            n_loci = int(haplotypes.shape[0])
        except Exception as e:
            raise ValueError("`haplotypes` must have shape (loci, samples, ancestries).") from e
        try:
            self._hap_n_samples = int(haplotypes.shape[1])
            self._hap_n_ancestries = int(haplotypes.shape[2])
        except Exception:
            self._hap_n_samples, self._hap_n_ancestries = 0, 0
        self.locus_method = "nearest"

        if self._preload_to_torch:
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
        for c, vgrp in vreset.groupby("chrom", sort=False, observed=True):
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
        mod = _get_array_module(block)

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
        if self._panel is not None:
            return self._fetch_hap_block_panel(np.asarray(v_idx, dtype=int))

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
        if self._preload_to_torch:
            device = self._hap_torch_cache.device if self._hap_torch_cache is not None else self._hap_torch_device
            return torch.empty((0, n_samp, n_anc), dtype=torch.float32, device=device)
        if self._panel is not None:
            return np.empty((0, n_samp, n_anc), dtype=np.float32)
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
def _to_pandas(df) -> pd.DataFrame | pd.Series:
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _get_sample_ids(df) -> List[str]:
    if hasattr(df, "to_arrow"):
        return df["sample_id"].to_arrow().to_pylist()
    return df["sample_id"].tolist()
