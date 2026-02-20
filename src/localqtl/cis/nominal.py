import time
import os, torch
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Optional

from ..utils import SimpleLogger
from ..stats import nominal_pvals_tensorqtl, get_t_pval
from ..iosinks import AsyncParquetSink
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, allele_stats, filter_by_maf
from ..regression_kernels import (
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .common import (
    residualize_matrix_with_covariates,
    residualize_batch,
    residualize_batch_interaction,
    prepare_interaction_covariate,
)

__all__ = [
    "map_nominal",
]

DEFAULT_ROW_GROUP_SIZE = 10_000_000

def _interaction_columns(n_ancestries: int) -> list[str]:
    cols: list[str] = []
    for anc in range(n_ancestries):
        suffix = f"anc{anc}"
        cols.extend([
            f"slope_gxh_{suffix}",
            f"slope_se_gxh_{suffix}",
            f"tstat_gxh_{suffix}",
            f"pval_gxh_{suffix}",
        ])
    return cols

def _nominal_parquet_schema(
        include_perm: bool,
        ancestry_model: str | None = None,
        n_ancestries: int | None = None,
) -> pa.Schema:
    fields = [
        pa.field("phenotype_id", pa.string()),
        pa.field("variant_id", pa.string()),
        pa.field("start_distance", pa.int32()),
        pa.field("end_distance", pa.int32()),
        pa.field("slope", pa.float32()),
        pa.field("slope_se", pa.float32()),
        pa.field("tstat", pa.float32()),
        pa.field("pval_nominal", pa.float32()),
        pa.field("af", pa.float32()),
        pa.field("ma_samples", pa.int32()),
        pa.field("ma_count", pa.int32()),
    ]
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        for col in _interaction_columns(int(n_ancestries)):
            fields.append(pa.field(col, pa.float32()))
    if include_perm:
        fields.append(pa.field("perm_max_r2", pa.float32()))
    return pa.schema(fields)


def _count_cis_pairs(ig, chrom: str | None = None) -> int:
    """Return the number of cis variant-phenotype pairs for a chromosome."""
    total = 0
    group_mode = getattr(ig, "group_s", None) is not None
    for batch in ig.generate_data(chrom=chrom):
        if not group_mode:
            if len(batch) in (4, 5):
                G_block = batch[1]
                n_phen = 1
            else:
                raise ValueError(f"Unexpected batch length for cis count: {len(batch)}")
        else:
            if len(batch) == 5:
                G_block = batch[1]
                ids = batch[3]
            elif len(batch) == 6:
                G_block = batch[1]
                ids = batch[4]
            else:
                raise ValueError(f"Unexpected grouped batch length for cis count: {len(batch)}")
            n_phen = len(ids)
        total += G_block.shape[0] * n_phen
    return int(total)


def _allocate_buffers(
        expected_columns,
        include_perm: bool,
        target_rows: int,
        interaction_columns: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    target = max(int(target_rows), 0)
    buffers: dict[str, np.ndarray] = {
        "phenotype_id":   np.empty(target, dtype=object),
        "variant_id":     np.empty(target, dtype=object),
        "start_distance": np.empty(target, dtype=np.int32),
        "end_distance":   np.empty(target, dtype=np.int32),
        "slope":           np.empty(target, dtype=np.float32),
        "slope_se":             np.empty(target, dtype=np.float32),
        "tstat":          np.empty(target, dtype=np.float32),
        "pval_nominal":   np.empty(target, dtype=np.float32),
        "af":             np.empty(target, dtype=np.float32),
        "ma_samples":     np.empty(target, dtype=np.int32),
        "ma_count":       np.empty(target, dtype=np.int32),
    }
    if interaction_columns:
        for col in interaction_columns:
            buffers[col] = np.empty(target, dtype=np.float32)
    if include_perm:
        buffers["perm_max_r2"] = np.empty(target, dtype=np.float32)
    return buffers


def _buffers_to_arrow(buffers: dict[str, np.ndarray], schema: pa.Schema, n_rows: int) -> pa.Table:
    arrays = []
    n = int(n_rows)
    for field in schema:
        data = buffers.get(field.name)
        if data is None:
            arrays.append(pa.nulls(n, type=field.type))
        else:
            arrays.append(pa.array(data[:n], type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _run_nominal_core(ig, variant_df, rez, nperm, device, maf_threshold: float = 0.0,
                      chrom: str | None = None, sink: "ParquetSink | None" = None,
                      target_rows: int | None = None, logger: SimpleLogger | None = None,
                      total_phenotypes: int | None = None,
                      ancestry_model: str | None = None,
                      n_ancestries: int | None = None,
                      interaction_covariate_t: torch.Tensor | None = None):
    """
    Shared inner loop for nominal (and optional permutation) mapping.
    Handles both InputGeneratorCis (no haps) and InputGeneratorCisWithHaps (haps).
    If sink is provided, stream out rows as Parquet and return None.
    """
    # Variant metadata
    idx_to_id = variant_df.index.to_numpy()
    pos_arr   = variant_df["pos"].to_numpy(np.int32)

    include_perm = nperm is not None and nperm > 0
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance",
        "end_distance", "slope", "slope_se", "tstat", "pval_nominal",
        "af", "ma_samples", "ma_count",
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
    if include_perm:
        expected_columns.append("perm_max_r2")

    if target_rows is None:
        if chrom is None:
            target_rows = _count_cis_pairs(ig, chrom=None)
        else:
            target_rows = _count_cis_pairs(ig, chrom)

    buffers = _allocate_buffers(expected_columns, include_perm, target_rows, interaction_columns)
    cursor = processed = 0
    variant_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    group_mode = getattr(ig, "group_s", None) is not None
    if total_phenotypes is None:
        total_phenotypes = ig.phenotype_df.shape[0]
    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)

    progress_interval = max(1, total_phenotypes // 10) if total_phenotypes else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"
    if include_perm and ancestry_model == "interaction":
        logger.write("  [warning] interaction mode with nperm>0: perm_max_r2 not computed")
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
        if H_block is not None:
            if isinstance(H_block, torch.Tensor):
                if H_block.device.type != device:
                    H_t = H_block.to(device=device, dtype=torch.float32)
                elif H_block.dtype != torch.float32:
                    H_t = H_block.to(dtype=torch.float32)
                else:
                    H_t = H_block
            else:
                H_t = torch.tensor(H_block, dtype=torch.float32, device=device)
        else:
            H_t = None

        # Impute / filter
        G_t, keep_mono, _ = impute_mean_and_filter(G_t)
        if G_t.shape[0] == 0:
            continue

        # Keep variant metadata / haps in sync with the monomorphic filter
        v_idx = v_idx[keep_mono.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mono]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1]

        # Optional MAF filter on the *current* (already-imputed/trimmed) G_t
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            if keep_maf.sum().item() == 0:
                continue
            # Apply MAF mask consistently across tensors/indices
            G_t   = G_t[keep_maf]
            v_idx = v_idx[keep_maf.detach().cpu().numpy()]
            if H_t is not None:
                H_t = H_t[keep_maf]

        # Sanity checks to catch any future drift
        assert G_t.shape[0] == v_idx.shape[0], "G_t and v_idx out of sync"
        if H_t is not None:
            assert H_t.shape[0] == G_t.shape[0], "G_t and H_t out of sync"

        # Allele statistics on imputed genotypes
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)
 
        # Residualize in one call; returns y as list in group mode
        interaction_mode = ancestry_model == "interaction" and H_t is not None
        covar_interaction = interaction_covariate_t is not None
        if interaction_mode:
            y_resid, G_resid, H_resid, I_resid = residualize_batch_interaction(
                P_list if group_mode else P_list[0], G_t, H_t, rez, center=True,
                group=group_mode
            )
        else:
            y_resid, G_resid, H_resid = residualize_batch(
                P_list if group_mode else P_list[0], G_t, H_t, rez, center=True,
                group=group_mode
            )
        y_iter = list(zip(y_resid, id_list)) if group_mode else [(y_resid, id_list[0])]

        # Variant metadata
        cache_key = v_idx.tobytes()
        if cache_key in variant_cache:
            var_ids, var_pos = variant_cache[cache_key]
        else:
            var_ids = idx_to_id[v_idx]
            var_pos = pos_arr[v_idx].astype(np.int32, copy=False)
            variant_cache[cache_key] = (var_ids, var_pos)
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Per-phenotype regressions in this window
        for y_t, pid in y_iter:
            gxh_slope = gxh_se = gxh_t = gxh_p = None
            if interaction_mode:
                pH = int(H_resid.shape[2])
                n_anc = int(n_ancestries) if n_ancestries is not None else pH
                H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid], dim=2)
                n = int(y_t.shape[0])
                p_pred = 1 + int(H_cov.shape[2])
                dof = max(n - int(k_eff) - p_pred, 1)

                gxh_slope = np.full((G_resid.shape[0], n_anc), np.nan, dtype=np.float32)
                gxh_se = np.full_like(gxh_slope, np.nan)
                gxh_t = np.full_like(gxh_slope, np.nan)
                gxh_p = np.full_like(gxh_slope, np.nan)

                for k in range(pH):
                    betas_k, ses_k, tstats_k = run_batch_regression(
                        y=y_t, G=I_resid[:, :, k], H=H_cov, k_eff=k_eff, device=device
                    )
                    gxh_slope[:, k] = betas_k[:, 0].detach().cpu().numpy()
                    gxh_se[:, k] = ses_k[:, 0].detach().cpu().numpy()
                    gxh_t[:, k] = tstats_k[:, 0].detach().cpu().numpy()
                    gxh_p[:, k] = get_t_pval(gxh_t[:, k], dof)

                best_k = np.nanargmax(np.abs(gxh_t[:, :pH]), axis=1)
                row_ix = np.arange(gxh_t.shape[0])
                slope = gxh_slope[row_ix, best_k]
                slope_se = gxh_se[row_ix, best_k]
                tstat = gxh_t[row_ix, best_k]
                pvals = gxh_p[row_ix, best_k]
                perm_max_r2 = np.nan if include_perm else None
            elif covar_interaction:
                c_t = interaction_covariate_t
                if c_t.device != G_resid.device:
                    c_t = c_t.to(G_resid.device)
                I_t = G_resid * c_t  # (m, n)
                c_rep = c_t.unsqueeze(0).expand(G_resid.shape[0], -1).unsqueeze(-1)
                if H_resid is None:
                    H_cov = torch.cat([G_resid.unsqueeze(-1), c_rep], dim=2)
                else:
                    H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid, c_rep], dim=2)
                n = int(y_t.shape[0])
                p_pred = 1 + int(H_cov.shape[2])
                dof = max(n - int(k_eff) - p_pred, 1)
                if include_perm:
                    perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
                    y_perm = torch.stack([y_t[idxp] for idxp in perms], dim=1) # (n x nperm)
                    betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                        y=y_t, G=I_t, H=H_cov, y_perm=y_perm, k_eff=k_eff, device=device
                    )
                    perm_max_r2 = r2_perm.max().item()
                else:
                    betas, ses, tstats = run_batch_regression(
                        y=y_t, G=I_t, H=H_cov, k_eff=k_eff, device=device
                    )
                    r2_perm = perm_max_r2 = None
                slope = betas[:, 0].detach().cpu().numpy()
                slope_se = ses[:, 0].detach().cpu().numpy()
                tstat = tstats[:, 0].detach().cpu().numpy()
                pvals = get_t_pval(tstat, dof)
            else:
                if include_perm:
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
            n_variants = var_ids.shape[0]

            # Pack and one GPU -> CPU transfer
            if interaction_mode or covar_interaction:
                floats = np.stack([slope, slope_se, tstat, pvals,
                                   af_t.detach().cpu().numpy()], axis=1).astype(np.float32)
                ints = np.stack([ma_samples_t.detach().cpu().numpy(),
                                 ma_count_t.detach().cpu().numpy()], axis=1).astype(np.int32)
            else:
                floats_t = torch.stack([betas[:,0], ses[:,0], tstats[:,0],
                                        pvals_t, af_t], dim=1).to(torch.float32)
                ints_t   = torch.stack([ma_samples_t, ma_count_t], dim=1).to(torch.int32)

                floats = floats_t.detach().to("cpu", non_blocking=True).numpy()
                ints   = ints_t.detach().to("cpu", non_blocking=True).numpy()

            row_slice = slice(cursor, cursor + n_variants)

            buffers["phenotype_id"][row_slice] = str(pid)
            buffers["variant_id"][row_slice] = var_ids
            np.subtract(var_pos, ig.phenotype_start[pid],
                        out=buffers["start_distance"][row_slice])
            np.subtract(var_pos, ig.phenotype_end[pid],
                        out=buffers["end_distance"][row_slice])

            beta_np, se_np, tstat_np, pval_np, af_np = (floats[:, i] for i in range(5))
            ma_samples_np, ma_count_np = ints[:, 0], ints[:, 1]

            buffers["slope"][row_slice] = beta_np
            buffers["slope_se"][row_slice] = se_np
            buffers["tstat"][row_slice] = tstat_np
            buffers["pval_nominal"][row_slice] = pval_np
            buffers["af"][row_slice] = af_np
            buffers["ma_samples"][row_slice] = ma_samples_np
            buffers["ma_count"][row_slice] = ma_count_np
            if interaction_mode and interaction_columns:
                for anc_idx in range(len(interaction_columns) // 4):
                    base = anc_idx * 4
                    buffers[interaction_columns[base + 0]][row_slice] = gxh_slope[:, anc_idx]
                    buffers[interaction_columns[base + 1]][row_slice] = gxh_se[:, anc_idx]
                    buffers[interaction_columns[base + 2]][row_slice] = gxh_t[:, anc_idx]
                    buffers[interaction_columns[base + 3]][row_slice] = gxh_p[:, anc_idx]

            if include_perm:
                fill_value = np.nan if perm_max_r2 is None else perm_max_r2
                buffers["perm_max_r2"][row_slice] = fill_value

            cursor += n_variants
            processed += 1
            if (
                    logger.verbose and total_phenotypes and progress_interval
                    and (processed % progress_interval == 0
                         or processed == total_phenotypes)
            ):
                logger.write(
                    f"      processed {processed}/{total_phenotypes} phenotypes on {chrom_label}"
                )

    if sink is not None:
        schema = _nominal_parquet_schema(include_perm, ancestry_model, n_ancestries)
        table = _buffers_to_arrow(buffers, schema, cursor)
        sink.write(table)
        return None

    if cursor == 0:
        return pd.DataFrame(columns=expected_columns)

    data = {col: buffers[col][:cursor] for col in expected_columns}
    return pd.DataFrame(data, columns=expected_columns)


def map_nominal(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        window: int = 1_000_000, nperm: Optional[int] = None, device: str = "cuda",
        out_dir: str = "./", out_prefix: str = "cis_nominal",
        compression: str = "snappy", return_df: bool = False,
        logger: SimpleLogger | None = None, verbose: bool = True,
        preload_haplotypes: bool = True, tensorqtl_flavor: bool = False,
        ancestry_model: str = "main",
        interaction_covariate: Optional[object] = None,
    ) -> pd.DataFrame:
    """
    Nominal cis-QTL scan with optional permutations and local ancestry.

    Adjusts for covariates by residualizing y, G, and H across samples using the
    same Residualizer (projection onto the orthogonal complement of C).

    Parameters
    ----------
    preload_haplotypes : bool, default True
        When haplotypes are provided, pre-load them into a contiguous torch.Tensor
        on the requested device to avoid per-batch host<->device transfers.
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)
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
    if ancestry_model == "interaction":
        logger.write("  * ancestry model: interaction (GxH)")
    if interaction_covariate is not None:
        logger.write("  * covariate interaction: GxC")
    if nperm is not None:
        logger.write(f"  * computing tensorQTL-style nominal p-values and {nperm:,} permutations")

    # Build the appropriate input generator
    ig = (
        InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
            loci_df=loci_df, group_s=group_s,
            preload_to_torch=(preload_haplotypes and haplotypes is not None),
            torch_device=torch_device,
        ) if haplotypes is not None else
        InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
    )

    # Residualize once using the filtered phenotypes from the generator
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)
    with logger.time_block(" Residualizing phenotypes", sync=sync):
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_use,
                                                          device, tensorqtl_flavor)

    ig.phenotype_df = pd.DataFrame(
        Y_resid.cpu().numpy(), index=ig.phenotype_df.index, columns=ig.phenotype_df.columns
    )

    # Pre-compute per-chromosome phenotype counts for logging
    phenotype_counts = ig.phenotype_pos_df['chr'].value_counts().to_dict()

    # Per-chromosome parquet streaming
    if ancestry_model == "interaction" and haplotypes is None:
        raise ValueError("ancestry_model='interaction' requires haplotypes.")
    if ancestry_model == "interaction" and interaction_covariate is not None:
        raise ValueError("interaction_covariate cannot be combined with ancestry_model='interaction'.")

    covariates_use = covariates_df
    if isinstance(interaction_covariate, str) and covariates_df is not None:
        if interaction_covariate in covariates_df.columns:
            covariates_use = covariates_df.drop(columns=[interaction_covariate])
    interaction_vec = prepare_interaction_covariate(
        interaction_covariate, covariates_df, phenotype_df.columns
    )
    interaction_cov_t = None
    if interaction_vec is not None:
        interaction_cov_t = torch.as_tensor(interaction_vec, dtype=torch.float32, device=device)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        include_perm = nperm is not None and nperm > 0
        n_anc = int(haplotypes.shape[2]) if haplotypes is not None else None
        schema = _nominal_parquet_schema(include_perm, ancestry_model, n_anc)
        with logger.time_block("Nominal scan (per-chrom streaming)", sync=sync, sec=False):
            for chrom in ig.chrs:
                out_path = os.path.join(out_dir, f"{out_prefix}.{chrom}.parquet")
                chrom_total = int(phenotype_counts.get(chrom, 0))
                if logger.verbose:
                    logger.write(f"    Mapping chromosome {chrom} ({chrom_total} phenotypes)")
                chrom_start = time.time()
                with logger.time_block(f"{chrom}: map_nominal", sync=sync):
                    target_rows = _count_cis_pairs(ig, chrom)
                    with AsyncParquetSink(
                            out_path, schema=schema, compression=compression,
                            row_group_size=DEFAULT_ROW_GROUP_SIZE,
                            use_dictionary=("phenotype_id",), write_statistics=False,
                            column_order=list(schema.names), max_queue_items=4,
                    ) as sink:
                        _run_nominal_core(
                            ig, variant_df, rez, nperm, device,
                            maf_threshold=maf_threshold, chrom=chrom, sink=sink,
                            target_rows=target_rows, logger=logger,
                            total_phenotypes=chrom_total,
                            ancestry_model=ancestry_model,
                            n_ancestries=n_anc,
                            interaction_covariate_t=interaction_cov_t,
                        )
                        # logger.write(f"{chrom}: ~{sink.rows:,} rows written")
                if logger.verbose:
                    elapsed = time.time() - chrom_start
                    logger.write(f"    Chromosome {chrom} completed in {elapsed:.2f}s")
        return None if not return_df else pd.DataFrame([])

    total_phenotypes = int(ig.phenotype_df.shape[0])
    if logger.verbose:
        logger.write(f"    Mapping all chromosomes ({total_phenotypes} phenotypes)")
    overall_start = time.time()
    with logger.time_block("Computing associations (nominal)", sync=sync):
        results = _run_nominal_core(ig, variant_df, rez, nperm, device,
                                    maf_threshold=maf_threshold,
                                    chrom=None, sink=None, logger=logger,
                                    total_phenotypes=total_phenotypes,
                                    ancestry_model=ancestry_model,
                                    n_ancestries=int(haplotypes.shape[2]) if haplotypes is not None else None,
                                    interaction_covariate_t=interaction_cov_t)
    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed nominal scan in {elapsed / 60:.2f} min")
    return results
