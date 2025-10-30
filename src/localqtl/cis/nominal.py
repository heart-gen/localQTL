import os, torch
import numpy as np
import pandas as pd
from typing import Optional

from ..utils import SimpleLogger
from ..iosinks import ParquetSink
from ..stats import nominal_pvals_tensorqtl
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, allele_stats, filter_by_maf
from ..regression_kernels import (
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .common import residualize_matrix_with_covariates, residualize_batch

__all__ = [
    "map_nominal",
]

def _run_nominal_core(ig, variant_df, rez, nperm, device, maf_threshold: float = 0.0,
                      chrom: str | None = None, sink: "ParquetSink | None" = None):
    """
    Shared inner loop for nominal (and optional permutation) mapping.
    Handles both InputGeneratorCis (no haps) and InputGeneratorCisWithHaps (haps).
    If sink is provided, stream out rows as Parquet and return None.
    """
    out_rows = []
    group_mode = getattr(ig, "group_s", None) is not None
    for batch in ig.generate_data(chrom=chrom):
        if not group_mode: # Ungrouped
            if len(batch) == 4:
                p, G_block, v_idx, pid = batch
                H_block = None
                P_list, id_list = [p], [pid]
            elif len(batch) == 5:
                p, G_block, v_idx, H_block, pid = batch
                P_list, id_list = [p], [pid]
            else:
                raise ValueError(f"Unexpected ungrouped batch length: len={len(batch)}")
        else: # Grouped
            if len(batch) == 5:
                P, G_block, v_idx, ids, _group_id = batch
                H_block = None
            elif len(batch) == 6:
                P, G_block, v_idx, H_block, ids, _group_id = batch
            else:
                raise ValueError(f"Unexpected grouped batch shape: len={len(batch)}")
            if isinstance(P, (list, tuple)):
                P_list = list(P)
            else:
                P = np.asarray(P)
                P_list = [P[i, :] for i in range(P.shape[0])] if P.ndim == 2 else [P]
            id_list = list(ids)

        # Tensors
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Impute / filter
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
            continue

        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue

        # Mask consistently
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1] 

        # Allele statistics on imputed genotypes
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)
 
        # Residualize in one call; returns y as list in group mode
        y_resid, G_resid, H_resid = residualize_batch(
            P_list if group_mode else P_list[0], G_t, H_t, rez, center=True,
            group=group_mode
        )
        y_iter = list(zip(y_resid, id_list)) if group_mode else [(y_resid, id_list[0])]

        # Variant metadata
        var_ids = variant_df.index.values[v_idx]
        var_pos = variant_df.iloc[v_idx]["pos"].values
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Per-phenotype regressions in this window
        for y_t, pid in y_iter:
            if nperm is not None:
                perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
                y_perm = torch.stack([y_t[idxp] for idxp in perms], dim=1) # (n x nperm)
                betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                    y=y_t, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                )
                pvals_t, _dof = nominal_pvals_tensorqtl(
                    y_t=y_t, G_resid=G_resid, H_resid=H_resid, k_eff=k_eff, tstats=tstats
                )
                perm_max_r2 = r2_perm.max().item()
            else:
                betas, ses, tstats = run_batch_regression(
                    y=y_t, G=G_resid, H=H_resid, k_eff=k_eff, device=device
                )
                pvals_t, _dof = nominal_pvals_tensorqtl(
                    y_t=y_t, G_resid=G_resid, H_resid=H_resid, k_eff=k_eff, tstats=tstats
                )
                r2_perm = perm_max_r2 = None

            # Distances to phenotype start/end
            start_pos = ig.phenotype_start[pid]
            end_pos = ig.phenotype_end[pid]
            start_distance = var_pos - start_pos
            end_distance = var_pos - end_pos

            # Assemble result rows
            df = pd.DataFrame({
                "phenotype_id": pid,
                "variant_id": var_ids,
                "pos": var_pos,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "beta": betas[:, 0].detach().cpu().numpy(),
                "se": ses[:, 0].detach().cpu().numpy(),
                "tstat": tstats[:, 0].detach().cpu().numpy(),
                "pval_nominal": pvals_t.detach().cpu().numpy(),
                "af": af_t.detach().cpu().numpy(),
                "ma_samples": ma_samples_t.detach().cpu().numpy(),
                "ma_count": ma_count_t.detach().cpu().numpy(),
            })
            if perm_max_r2 is not None:
                df["perm_max_r2"] = perm_max_r2

            if sink is None:
                out_rows.append(df)
            else:
                sink.write(df)

    if sink is None:
        return pd.concat(out_rows, axis=0).reset_index(drop=True)
    return None


def map_nominal(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        window: int = 1_000_000, nperm: Optional[int] = None, device: str = "cuda",
        out_dir: str = "./", out_prefix: str = "cis_nominal",
        compression: str = "snappy", return_df: bool = False,
        logger: SimpleLogger | None = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Nominal cis-QTL scan with optional permutations and local ancestry.

    Adjusts for covariates by residualizing y, G, and H across samples using the
    same Residualizer (projection onto the orthogonal complement of C).
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
    sync = (torch.cuda.synchronize if device == "cuda" else None)

    # Header (tensorQTL-style)
    logger.write("cis-QTL mapping: nominal associations for all variant–phenotype pairs")
    logger.write(f"  * device: {device}")
    logger.write(f"  * {phenotype_df.shape[1]} samples")
    logger.write(f"  * {phenotype_df.shape[0]} phenotypes")
    logger.write(f"  * {variant_df.shape[0]} variants")
    logger.write(f"  * cis-window: \u00B1{window:,}")
    if maf_threshold and maf_threshold > 0:
        logger.write(f"  * applying in-sample {maf_threshold:g} MAF filter")
    if covariates_df is not None:
        logger.write(f"  * {covariates_df.shape[1]} covariates")
    if haplotypes is not None:
        K = int(haplotypes.shape[2])
        logger.write(f"  * including local ancestry channels (K={K})")
    if nperm is not None:
        logger.write(f"  * computing tensorQTL-style nominal p-values and {nperm:,} permutations")

    # Build the appropriate input generator
    ig = (
        InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
            loci_df=loci_df, group_s=group_s) if haplotypes is not None else
        InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
    )

    # Residualize once using the filtered phenotypes from the generator
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)
    with logger.time_block("Residualizing phenotypes", sync=sync):
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_df, device)

    ig.phenotype_df = pd.DataFrame(
        Y_resid.cpu().numpy(), index=ig.phenotype_df.index, columns=ig.phenotype_df.columns
    )

    # Per-chromosome parquet streaming
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with logger.time_block("Nominal scan (per-chrom streaming)", sync=sync):
            for chrom in ig.chrs:
                out_path = os.path.join(out_dir, f"{out_prefix}.chr{chrom}.parquet")
                with logger.time_block(f"chr{chrom}: map_nominal", sync=sync):
                    with ParquetSink(out_path, compression=compression) as sink:
                        _run_nominal_core(
                            ig, variant_df, rez, nperm, device,
                            maf_threshold=maf_threshold,
                            chrom=chrom,
                            sink=sink,
                        )
                    logger.write(f"chr{chrom}: wrote {sink.rows:,} rows -> {out_path}")
        return None if not return_df else pd.DataFrame([])

    with logger.time_block("Computing associations (nominal)", sync=sync):
        return _run_nominal_core(ig, variant_df, rez, nperm, device,
                                 maf_threshold=maf_threshold,
                                 chrom=None, sink=None)
    
