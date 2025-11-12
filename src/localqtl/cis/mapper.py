import torch
import numpy as np
import pandas as pd
from typing import Mapping, Optional, Sequence, Union

from ..utils import SimpleLogger, pick_device
from ..stats import calculate_qvalues as _calculate_qvalues
from .nominal import map_nominal as _map_nominal
from .permutations import map_permutations as _map_permutations
from .independent import map_independent as _map_independent
from .postproc import get_significant_pairs as _get_significant_pairs

__all__ = [
    "CisMapper",
]

class CisMapper:
    """
    Thin convenience wrapper that delegates to the functional APIs:
      - map_nominal
      - map_permutations
      - map_independent
    """

    def __init__(
            self,
            genotype_df: pd.DataFrame,
            variant_df: pd.DataFrame,
            phenotype_df: pd.DataFrame,
            phenotype_pos_df: pd.DataFrame,
            covariates_df: Optional[pd.DataFrame] = None,
            group_s: Optional[pd.Series] = None,
            haplotypes: Optional[object] = None,
            loci_df: Optional[pd.DataFrame] = None,
            device: str = "auto",
            window: int = 1_000_000,
            maf_threshold: float = 0.0,
            out_dir: str = "./",
            out_prefix: str = "cis_nominal",
            compression: str = "snappy",
            return_df: bool = False,
            tensorqtl_flavor: bool = False,
            logger: Optional[SimpleLogger] = None,
            verbose: bool = True,
    ):
        # Store inputs (no IG construction or residualization here)
        self.genotype_df = genotype_df
        self.variant_df = variant_df
        self.phenotype_df = phenotype_df
        self.phenotype_pos_df = phenotype_pos_df
        self.covariates_df = covariates_df
        self.group_s = group_s
        self.haplotypes = haplotypes
        self.loci_df = loci_df

        self.device = pick_device(device)
        self.window = window
        self.maf_threshold = maf_threshold
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.compression = compression
        self.return_df = return_df
        self.tensorqtl_flavor = tensorqtl_flavor

        self.logger = logger or SimpleLogger(verbose=verbose, timestamps=True)

    # ----------------------------
    # Delegating methods
    # ----------------------------
    def map_nominal(self, nperm: Optional[int] = None,
                    maf_threshold: Optional[float] = None) -> pd.DataFrame:
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return _map_nominal(
            genotype_df=self.genotype_df,
            variant_df=self.variant_df,
            phenotype_df=self.phenotype_df,
            phenotype_pos_df=self.phenotype_pos_df,
            covariates_df=self.covariates_df,
            haplotypes=self.haplotypes,
            loci_df=self.loci_df,
            group_s=self.group_s,
            maf_threshold=mt,
            window=self.window,
            nperm=nperm,
            device=self.device,
            out_dir=self.out_dir,
            out_prefix=self.out_prefix,
            compression=self.compression,
            return_df=self.return_df,
            logger=self.logger,
            tensorqtl_flavor=self.tensorqtl_flavor,
            verbose=getattr(self.logger, "verbose", True),
        )

    def map_permutations(
            self, nperm: int = 10_000, beta_approx: bool = True,
            perm_chunk: int = 4096, maf_threshold: Optional[float] = None,
            seed: Optional[int] = None
    ) -> pd.DataFrame:
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return _map_permutations(
            genotype_df=self.genotype_df,
            variant_df=self.variant_df,
            phenotype_df=self.phenotype_df,
            phenotype_pos_df=self.phenotype_pos_df,
            covariates_df=self.covariates_df,
            haplotypes=self.haplotypes,
            loci_df=self.loci_df,
            group_s=self.group_s,
            maf_threshold=mt,
            window=self.window,
            nperm=nperm,
            device=self.device,
            beta_approx=beta_approx,
            perm_chunk=perm_chunk,
            seed=seed,
            logger=self.logger,
            tensorqtl_flavor=self.tensorqtl_flavor,
            verbose=getattr(self.logger, "verbose", True),
        )

    def map_independent(
            self, cis_df: pd.DataFrame, fdr: float = 0.05, fdr_col: str = "qval",
            nperm: int = 10_000, perm_chunk: int = 4096,
            random_tiebreak: bool = False, missing_val: float = -9.0,
            beta_approx: bool = True, seed: Optional[int] = None,
            maf_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return _map_independent(
            genotype_df=self.genotype_df,
            variant_df=self.variant_df,
            cis_df=cis_df,
            phenotype_df=self.phenotype_df, # independent expects RAW
            phenotype_pos_df=self.phenotype_pos_df,
            covariates_df=self.covariates_df,
            haplotypes=self.haplotypes,
            loci_df=self.loci_df,
            group_s=self.group_s,
            maf_threshold=mt,
            fdr=fdr,
            fdr_col=fdr_col,
            nperm=nperm,
            window=self.window,
            missing=missing_val,
            random_tiebreak=random_tiebreak,
            device=self.device,
            beta_approx=beta_approx,
            perm_chunk=perm_chunk,
            seed=seed,
            logger=self.logger,
            tensorqtl_flavor=self.tensorqtl_flavor,
            verbose=getattr(self.logger, "verbose", True),
        )

    def calculate_qvalues(
            self, perm_df: pd.DataFrame, fdr: float = 0.05,
            qvalue_lambda: Optional[float] = None,
    ) -> pd.DataFrame:
        """Annotate permutation results with q-values via :func:`calculate_qvalues`."""
        return _calculate_qvalues(
            perm_df, fdr=fdr,
            qvalue_lambda=qvalue_lambda,
            logger=self.logger,
        )

    def get_significant_pairs(
            self,
            res_df: pd.DataFrame,
            nominal_files: Optional[Union[str, Mapping[str, str]]] = None,
            group_s: Optional[pd.Series] = None,
            fdr: float = 0.05,
            columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Thin wrapper around :func:`get_significant_pairs` using stored configuration."""
        gs = self.group_s if group_s is None else group_s
        if nominal_files is None:
            out_dir = self.out_dir.rstrip("/")
            if not out_dir:
                out_dir = "."
            nominal_files = f"{out_dir}/{self.out_prefix}*.parquet"
        return _get_significant_pairs(
            res_df=res_df,
            nominal_files=nominal_files,
            group_s=gs,
            fdr=fdr,
            columns=columns,
            logger=self.logger,
        )
