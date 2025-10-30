import os, torch
import numpy as np
import pandas as pd
from typing import Optional

from ..utils import SimpleLogger
from ..iosinks import ParquetSink
from ..regression_kernels import Residualizer
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps

from .nominal import map_nominal
from .independent import map_independent
from .permutations import map_permutations

class CisMapper:
    """
    Convenience wrapper to run analysis.
    """
    def __init__(
            self, genotype_df: pd.DataFrame, variant_df: pd.DataFrame,
            phenotype_df: pd.DataFrame, phenotype_pos_df: pd.DataFrame,
            covariates_df: Optional[pd.DataFrame] = None,
            group_s: Optional[pd.Series] = None,
            haplotypes: Optional[object] = None,
            loci_df: Optional[pd.DataFrame] = None,
            device: str = "auto", window: int = 1_000_000,
            maf_threshold: float = 0.0,
            out_dir: Optional[str] = None, out_prefix: str = "cis_nominal",
            compression: str = "snappy", return_df: bool = True,
            logger: SimpleLogger | None = None, verbose: bool = True
    ):
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available()) else
                       device if device in ("cuda", "cpu") else "cpu")
        self.logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
        self.variant_df = variant_df
        self.window = window
        self.maf_threshold = maf_threshold
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.compression = compression
        self.return_df = return_df

        # Keep RAW phenotypes and covariates for independent mapping
        self.phenotype_df_raw = phenotype_df.copy()                 # keep RAW phenotypes
        self.covariates_df = covariates_df.copy() if covariates_df is not None else None

        self.ig = (
            InputGeneratorCisWithHaps(
                genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
                loci_df=loci_df, group_s=group_s) if haplotypes is not None else
            InputGeneratorCis(
                genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
        )

        # Record sample order
        self.sample_order = self.ig.phenotype_df.columns.tolist()

        # Residualize phenotypes once for nominal/permutation scans
        Y = torch.tensor(self.ig.phenotype_df.values, dtype=torch.float32, device=self.device)
        sync = (torch.cuda.synchronize if self.device == "cuda" else None)
        with self.logger.time_block("Residualizing phenotypes", sync=sync):
            Y_resid, self.rez = residualize_matrix_with_covariates(Y, covariates_df, self.device)
        self.ig.phenotype_df = pd.DataFrame(Y_resid.cpu().numpy(),
                                            index=self.ig.phenotype_df.index,
                                            columns=self.ig.phenotype_df.columns)

        # Header
        self.logger.write("CisMapper initialized")
        self.logger.write(f"  * device: {self.device}")
        self.logger.write(f"  * phenotypes: {self.ig.phenotype_df.shape[0]}")
        self.logger.write(f"  * samples: {self.ig.phenotype_df.shape[1]}")
        self.logger.write(f"  * variants: {self.variant_df.shape[0]}")
        self.logger.write(f"  * cis-window: \u00B1{self.window:,}")
        if self.maf_threshold and self.maf_threshold > 0:
            self.logger.write(f"  * MAF filter: {self.maf_threshold:g}")
        if hasattr(self.ig, "haplotypes") and self.ig.haplotypes is not None:
            self.logger.write(f"  * local ancestry channels (K={int(self.ig.haplotypes.shape[2])})")

    def map_nominal(self, nperm: int | None = None, maf_threshold: float | None = None) -> pd.DataFrame:
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        sync = (torch.cuda.synchronize if self.device == "cuda" else None)
        # Per-chromosome parquet streaming
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            with self.logger.time_block("Nominal scan (per-chrom streaming)", sync=sync):
                for chrom in self.ig.chrs:
                    out_path = os.path.join(self.out_dir, f"{self.out_prefix}.chr{chrom}.parquet")
                    with self.logger.time_block(f"chr{chrom}: map_nominal", sync=sync):
                        with ParquetSink(out_path, compression=compression) as sink:
                            _run_nominal_core(
                                self.ig, self.variant_df, self.rez, nperm, self.device,
                                maf_threshold=mt, chrom=chrom, sink=sink,
                            )
                        self.logger.write(f"chr{chrom}: wrote {sink.rows:,} rows -> {out_path}")
            return None if not return_df else pd.DataFrame([])

        with self.logger.time_block("Computing associations (nominal)", sync=sync):
            return _run_nominal_core(self.ig, self.variant_df, self.rez, nperm,
                                     self.device, maf_threshold=mt)

    def map_permutations(self, nperm: int=10_000, beta_approx: bool=True,
                         maf_threshold: float | None = None) -> pd.DataFrame:
        """Empirical cis-QTLs (top per phenotype) with permutation p-values."""
        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        sync = (torch.cuda.synchronize if self.device == "cuda" else None)
        with self.logger.time_block("Computing associations (permutations)", sync=sync):
            core = _run_permutation_core_group if getattr(self.ig, "group_s", None) is not None else _run_permutation_core
            return core(self.ig, self.variant_df, self.rez, nperm=nperm, device=self.device,
                        beta_approx=beta_approx, maf_threshold=mt)

    def map_independent(self, cis_df: pd.DataFrame, fdr: float = 0.05,
                        fdr_col: str = "qval", nperm: int = 10_000, 
                        maf_threshold: float | None = None,
                        random_tiebreak: bool = False, seed: int | None = None,
                        missing_val: float = -9.0, beta_approx: bool = True) -> pd.DataFrame:
        """
        Forward–backward conditional cis-QTLs, seeded from FDR-significant rows in `cis_df`.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        mt = self.maf_threshold if maf_threshold is None else maf_threshold
        return map_independent(
            genotype_df=self.ig.genotype_df,
            variant_df=self.variant_df,
            cis_df=cis_df,
            phenotype_df=self.phenotype_df_raw,
            phenotype_pos_df=self.ig.phenotype_pos_df,
            covariates_df=self.covariates_df,
            haplotypes=getattr(self.ig, "haplotypes", None),
            loci_df=getattr(self.ig, "loci_df", None),
            group_s=getattr(self.ig, "group_s", None),
            maf_threshold=mt,
            fdr=fdr, fdr_col=fdr_col, nperm=nperm,
            window=self.window, missing=missing_val,
            random_tiebreak=random_tiebreak, device=self.device,
            beta_approx=beta_approx,
            logger=self.logger, verbose=self.logger.verbose if hasattr(self.logger, "verbose") else True,
        )

    def _build_residualizer_aug(self, extra_cols: list[np.ndarray]) -> Residualizer:
        """
        Build a Residualizer for [baseline covariates || accepted lead-variant dosages].
        """
        if self.covariates_df is None:
            C = np.empty((len(self.sample_order), 0), dtype=np.float32)
        else:
            C = self.covariates_df.loc[self.sample_order].to_numpy(dtype=np.float32, copy=False)

        if extra_cols:
            Xadd = np.column_stack([np.asarray(x, dtype=np.float32) for x in extra_cols])
            C = np.hstack([C, Xadd]).astype(np.float32, copy=False)

        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        return Residualizer(C_t)

    # Helper functions
    def _residualize(self, Y, C):
        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        CtC_inv = torch.linalg.inv(C_t.T @ C_t)
        proj = C_t @ (CtC_inv @ (C_t.T @ Y_t))
        return (Y_t - proj).cpu().numpy()

    def _regress(self, X, y):
        # closed-form OLS
        XtX = X.T @ X
        XtX_inv = torch.linalg.inv(XtX)
        betas = XtX_inv @ (X.T @ y)
        y_hat = X @ betas
        resid = y - y_hat
        k_eff = self.rez.Q_t.shape[1] if self.rez is not None else 0
        p = X.shape[-1]
        dof = X.shape[0] - k_eff - p
        sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof
        se = torch.sqrt(torch.diag(XtX_inv) * sigma2)
        tstats = betas / se
        return tstats, betas, se
