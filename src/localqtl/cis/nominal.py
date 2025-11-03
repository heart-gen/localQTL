import time
import os, torch
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Optional

from ..utils import SimpleLogger
from ..stats import nominal_pvals_tensorqtl
from ..iosinks import AsyncParquetSink
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

DEFAULT_ROW_GROUP_SIZE = 10_000_000

def _nominal_parquet_schema(include_perm: bool) -> pa.Schema:
    fields = [
        pa.field("phenotype_id", pa.string()),
        pa.field("variant_id", pa.string()),
        pa.field("pos", pa.int32()),
        pa.field("start_distance", pa.int32()),
        pa.field("end_distance", pa.int32()),
        pa.field("beta", pa.float32()),
        pa.field("se", pa.float32()),
        pa.field("tstat", pa.float32()),
        pa.field("pval_nominal", pa.float32()),
        pa.field("af", pa.float32()),
        pa.field("ma_samples", pa.int32()),
        pa.field("ma_count", pa.int32()),
    ]
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


def _allocate_buffers(expected_columns, include_perm: bool, target_rows: int) -> dict[str, np.ndarray]:
    target = max(int(target_rows), 0)
    buffers: dict[str, np.ndarray] = {
        "phenotype_id":   np.empty(target, dtype=object),
        "variant_id":     np.empty(target, dtype=object),
        "pos":            np.empty(target, dtype=np.int32),
        "start_distance": np.empty(target, dtype=np.int32),
        "end_distance":   np.empty(target, dtype=np.int32),
        "beta":           np.empty(target, dtype=np.float32),
        "se":             np.empty(target, dtype=np.float32),
        "tstat":          np.empty(target, dtype=np.float32),
        "pval_nominal":   np.empty(target, dtype=np.float32),
        "af":             np.empty(target, dtype=np.float32),
        "ma_samples":     np.empty(target, dtype=np.int32),
        "ma_count":       np.empty(target, dtype=np.int32),
    }
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
                      total_phenotypes: int | None = None):
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
        "phenotype_id", "variant_id", "pos", "start_distance",
        "end_distance", "beta", "se", "tstat", "pval_nominal",
        "af", "ma_samples", "ma_count",
    ]
    if include_perm:
        expected_columns.append("perm_max_r2")

    if target_rows is None:
        if chrom is None:
            target_rows = _count_cis_pairs(ig, chrom=None)
        else:
            target_rows = _count_cis_pairs(ig, chrom)

    buffers = _allocate_buffers(expected_columns, include_perm, target_rows)
    cursor = processed = 0
    variant_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    group_mode = getattr(ig, "group_s", None) is not None
    if total_phenotypes is None:
        total_phenotypes = ig.phenotype_df.shape[0]
    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)

    progress_interval = max(1, total_phenotypes // 10) if total_phenotypes else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"
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
            floats_t = torch.stack([betas[:,0], ses[:,0], tstats[:,0],
                                    pvals_t, af_t], dim=1).to(torch.float32)
            ints_t   = torch.stack([ma_samples_t, ma_count_t], dim=1).to(torch.int32)

            floats = floats_t.detach().to("cpu", non_blocking=True).numpy()
            ints   = ints_t.detach().to("cpu", non_blocking=True).numpy()

            row_slice = slice(cursor, cursor + n_variants)

            buffers["phenotype_id"][row_slice] = str(pid)
            buffers["variant_id"][row_slice] = var_ids
            buffers["pos"][row_slice] = var_pos
            np.subtract(var_pos, ig.phenotype_start[pid],
                        out=buffers["start_distance"][row_slice])
            np.subtract(var_pos, ig.phenotype_end[pid],
                        out=buffers["end_distance"][row_slice])

            beta_np, se_np, tstat_np, pval_np, af_np = (floats[:, i] for i in range(5))
            ma_samples_np, ma_count_np = ints[:, 0], ints[:, 1]

            buffers["beta"][row_slice] = beta_np
            buffers["se"][row_slice] = se_np
            buffers["tstat"][row_slice] = tstat_np
            buffers["pval_nominal"][row_slice] = pval_np
            buffers["af"][row_slice] = af_np
            buffers["ma_samples"][row_slice] = ma_samples_np
            buffers["ma_count"][row_slice] = ma_count_np

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
        schema = _nominal_parquet_schema(include_perm)
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
        preload_haplotypes: bool = True,
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
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_df, device)

    ig.phenotype_df = pd.DataFrame(
        Y_resid.cpu().numpy(), index=ig.phenotype_df.index, columns=ig.phenotype_df.columns
    )

    # Pre-compute per-chromosome phenotype counts for logging
    phenotype_counts = ig.phenotype_pos_df['chr'].value_counts().to_dict()

    # Per-chromosome parquet streaming
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        include_perm = nperm is not None and nperm > 0
        schema = _nominal_parquet_schema(include_perm)
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
                                    total_phenotypes=total_phenotypes)
    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed nominal scan in {elapsed / 60:.2f} min")
    return results
