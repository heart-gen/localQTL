import torch
import numpy as np
import pandas as pd
from typing import Optional

from ..utils import SimpleLogger, pick_device
from ..stats import calculate_qvalues as _calculate_qvalues
from .nominal import map_nominal as _map_nominal
from .permutations import map_permutations as _map_permutations
from .independent import map_independent as _map_independent

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
            verbose=getattr(self.logger, "verbose", True),
        )

    def map_permutations(
            self, nperm: int = 10_000, beta_approx: bool = True,
            maf_threshold: Optional[float] = None,
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
            seed=seed,
            logger=self.logger,
            verbose=getattr(self.logger, "verbose", True),
        )

    def map_independent(
            self, cis_df: pd.DataFrame, fdr: float = 0.05, fdr_col: str = "qval",
            nperm: int = 10_000, maf_threshold: Optional[float] = None,
            random_tiebreak: bool = False, seed: Optional[int] = None,
            missing_val: float = -9.0, beta_approx: bool = True,
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
            seed=seed,
            logger=self.logger,
            verbose=getattr(self.logger, "verbose", True),
        )

    def calculate_qvalues(
            self, perm_df: pd.DataFrame, fdr: float = 0.05,
            qvalue_lambda: Optional[float] = None,
    ) -> pd.DataFrame:
        """Annotate permutation results with q-values via :func:`calculate_qvalues`."""
        return _calculate_qvalues(
            perm_df,
            fdr=fdr,
            qvalue_lambda=qvalue_lambda,
            logger=self.logger,
        )
