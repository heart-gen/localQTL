import time
import os, torch
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Optional

from ..utils import SimpleLogger
from ..stats import nominal_pvals_tensorqtl, get_t_pval, get_f_pval
from ..iosinks import AsyncParquetSink
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import (impute_mean_and_filter, allele_stats, filter_by_maf, ancestry_allele_stats,
                       filter_maf_interaction, interaction_half_mask)
from ..regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_interaction_regression,
    run_batch_dosage_regression,
    run_batch_regression_with_permutations,
)
from .common import (
    residualize_matrix_with_covariates,
    residualize_batch,
    residualize_batch_interaction,
    residualize_batch_dosage,
    prepare_interaction_covariate,
    validate_ancestry_model,
    project_interaction,
)

__all__ = [
    "map_nominal",
    "map_nominal_loci",
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


def _dosage_columns(n_ancestries: int) -> list[str]:
    """
    Columns of the ancestry-specific dosage model (``ancestry_model='dosage'``):
    per-ancestry effects, the joint test, pairwise differences, then the
    per-ancestry in-sample allele statistics (``an``: allele number, ``ac``:
    counted-allele count, ``af``: ``ac/an``, ``ma_count``: minor-allele count
    within the ancestry; fractional for posterior-expected ancestry counts).
    """
    cols: list[str] = []
    for anc in range(n_ancestries):
        cols.extend([f"slope_anc{anc}", f"slope_se_anc{anc}", f"tstat_anc{anc}", f"pval_anc{anc}"])
    cols.extend(["fstat_joint", "pval_joint"])
    for i in range(n_ancestries):
        for j in range(i + 1, n_ancestries):
            cols.extend([f"z_diff_anc{i}_anc{j}", f"pval_diff_anc{i}_anc{j}"])
    for anc in range(n_ancestries):
        cols.extend([f"an_anc{anc}", f"ac_anc{anc}", f"af_anc{anc}", f"ma_count_anc{anc}"])
    return cols


def _nominal_parquet_schema(
        include_perm: bool,
        ancestry_model: str | None = None,
        n_ancestries: int | None = None,
        covariate_interaction: bool = False,
        interaction_names: list[str] | None = None,
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
    if ancestry_model == "dosage" and n_ancestries is not None and n_ancestries > 0:
        for col in _dosage_columns(int(n_ancestries)):
            fields.append(pa.field(col, pa.float32()))
    if covariate_interaction:
        for col in _covariate_interaction_columns(interaction_names):
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


def _covariate_interaction_columns(names: list[str] | None = None) -> list[str]:
    """
    Columns of the ``y ~ g + i + g:i`` covariate-interaction model.  With one
    interaction term the canonical ``b_g/se_g/tstat_g/pval_g``, ``b_i...``,
    ``b_gi...`` names are used; with several terms the variable names are
    substituted (tensorQTL convention): ``b_<name>``, ``se_<name>``,
    ``tstat_<name>``, ``pval_<name>`` for the main effects and ``b_g-<name>``
    ... for the genotype x term interactions.
    """
    if names is None or len(names) <= 1:
        return [
            "b_g", "se_g", "tstat_g", "pval_g",
            "b_i", "se_i", "tstat_i", "pval_i",
            "b_gi", "se_gi", "tstat_gi", "pval_gi",
        ]
    cols = ["b_g", "se_g", "tstat_g", "pval_g"]
    for v in names:
        cols.extend([f"b_{v}", f"se_{v}", f"tstat_{v}", f"pval_{v}"])
    for v in names:
        cols.extend([f"b_g-{v}", f"se_g-{v}", f"tstat_g-{v}", f"pval_g-{v}"])
    return cols


def _interaction_term_columns(names: list[str] | None, k: int) -> tuple[str, str]:
    """Suffixes of the main-effect and interaction columns of term ``k`` (``("i", "gi")`` or names)."""
    if names is None or len(names) <= 1:
        return "i", "gi"
    return str(names[k]), f"g-{names[k]}"


def _run_nominal_core(ig, variant_df, rez, nperm, device, maf_threshold: float = 0.0,
                      chrom: str | None = None, sink: "AsyncParquetSink | None" = None,
                      target_rows: int | None = None, logger: SimpleLogger | None = None,
                      total_phenotypes: int | None = None,
                      ancestry_model: str | None = None,
                      n_ancestries: int | None = None,
                      interaction_covariate_t: torch.Tensor | None = None,
                      min_ancestry_ac: float = 0.0,
                      interaction_names: list[str] | None = None,
                      interaction_mask_t: torch.Tensor | None = None,
                      maf_threshold_interaction: float = 0.0,
                      paired_ctx: dict | None = None,
                      top_rows: list | None = None,
                      run_eigenmt: bool = False,
                      collect_pairs: bool = True):
    """
    Shared inner loop for nominal (and optional permutation) mapping.
    Handles both InputGeneratorCis (no haps) and InputGeneratorCisWithHaps (haps).
    If sink is provided, stream out rows as Parquet and return None.

    ``interaction_covariate_t`` is ``(n,)`` or ``(n, ni)``; ``paired_ctx`` maps
    phenotype IDs to ``(Residualizer, residualized y)`` for phenotype-specific
    covariates; ``top_rows`` (a list) collects the top interaction association
    per phenotype / group; ``collect_pairs=False`` skips the pair-level output.
    """
    # Variant metadata
    idx_to_id = variant_df.index.to_numpy()
    pos_arr   = variant_df["pos"].to_numpy(np.int32)

    include_perm = nperm is not None and nperm > 0
    covar_interaction = interaction_covariate_t is not None
    ni = 0
    if covar_interaction:
        ni = 1 if interaction_covariate_t.dim() == 1 else int(interaction_covariate_t.shape[1])
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance",
        "end_distance", "slope", "slope_se", "tstat", "pval_nominal",
        "af", "ma_samples", "ma_count",
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
    dosage_columns: list[str] = []
    if ancestry_model == "dosage" and n_ancestries is not None and n_ancestries > 0:
        dosage_columns = _dosage_columns(int(n_ancestries))
        expected_columns.extend(dosage_columns)
    covar_int_columns: list[str] = []
    if covar_interaction:
        covar_int_columns = _covariate_interaction_columns(interaction_names)
        expected_columns.extend(covar_int_columns)
    if include_perm:
        expected_columns.append("perm_max_r2")

    if target_rows is None and collect_pairs:
        if chrom is None:
            target_rows = _count_cis_pairs(ig, chrom=None)
        else:
            target_rows = _count_cis_pairs(ig, chrom)

    all_extra_cols = interaction_columns + dosage_columns + covar_int_columns
    buffers = (_allocate_buffers(expected_columns, include_perm, target_rows, all_extra_cols)
               if collect_pairs else None)
    cursor = processed = 0
    variant_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    group_mode = getattr(ig, "group_s", None) is not None
    if total_phenotypes is None:
        total_phenotypes = ig.phenotype_df.shape[0]
    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    if run_eigenmt:
        from ..eigenmt import compute_tests

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
            group_id = None
        else: # Grouped
            if len(batch) == 5:
                P, G_block, v_idx, ids, group_id = batch
                H_block = None
            elif len(batch) == 6:
                P, G_block, v_idx, H_block, ids, group_id = batch
            else:
                raise ValueError(f"Unexpected grouped batch shape: len={len(batch)}")
            if isinstance(P, (list, tuple)):
                P_list = list(P)
            else:
                P = np.asarray(P)
                P_list = [P[i, :] for i in range(P.shape[0])] if P.ndim == 2 else [P]
            id_list = list(ids)

        # Phenotype-specific (paired) covariates: swap in that phenotype's residualizer
        rez_b = rez
        if paired_ctx and not group_mode and id_list[0] in paired_ctx:
            rez_b, y_paired = paired_ctx[id_list[0]]
            P_list = [y_paired]

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
            # The dosage model needs every ancestry channel (sum_k D_k = G);
            # the other models drop the last one for identifiability.
            if H_t.shape[2] > 1 and ancestry_model != "dosage":
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

        # Interaction MAF filter (both halves of the interaction variable)
        if covar_interaction and (interaction_mask_t is not None or maf_threshold_interaction > 0):
            G_t, keep_int = filter_maf_interaction(G_t, interaction_mask_t, maf_threshold_interaction)
            if G_t.shape[0] == 0:
                continue
            v_idx = v_idx[keep_int.detach().cpu().numpy()]
            if H_t is not None:
                H_t = H_t[keep_int]

        # Sanity checks to catch any future drift
        assert G_t.shape[0] == v_idx.shape[0], "G_t and v_idx out of sync"
        if H_t is not None:
            assert H_t.shape[0] == G_t.shape[0], "G_t and H_t out of sync"

        # Allele statistics on imputed genotypes
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)
        tests_emt = compute_tests(G_t) if (run_eigenmt and covar_interaction and top_rows is not None) else None

        # Residualize in one call; returns y as list in group mode
        interaction_mode = ancestry_model == "interaction" and H_t is not None
        dosage_mode = ancestry_model == "dosage" and H_t is not None
        # Per-ancestry allele statistics from the raw (imputed, un-residualized) block
        anc_stats_np: dict[str, np.ndarray] | None = None
        if dosage_mode:
            anc_stats_np = {k: v.detach().cpu().numpy().astype(np.float32)
                            for k, v in ancestry_allele_stats(G_t, H_t).items()}
        if interaction_mode:
            y_resid, G_resid, H_resid, I_resid = residualize_batch_interaction(
                P_list if group_mode else P_list[0], G_t, H_t, rez_b, center=True,
                group=group_mode
            )
        elif dosage_mode:
            y_resid, D_resid, L_resid = residualize_batch_dosage(
                P_list if group_mode else P_list[0], G_t, H_t, rez_b, center=True,
                group=group_mode
            )
            G_resid, H_resid = G_t, None
        else:
            y_resid, G_resid, H_resid = residualize_batch(
                P_list if group_mode else P_list[0], G_t, H_t, rez_b, center=True,
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
        k_eff = rez_b.k_eff if rez_b is not None else 1

        # Covariate-interaction design shared by the phenotypes of this window
        if covar_interaction:
            c_t = interaction_covariate_t
            if c_t.device != G_resid.device:
                c_t = c_t.to(G_resid.device)
            c2 = c_t.unsqueeze(-1) if c_t.dim() == 1 else c_t                   # (n, ni)
            # i and g:i (built from the raw genotypes) enter the design residualized
            # like g (Frisch–Waugh), as in tensorQTL
            i_res_t = project_interaction(c2.T.contiguous(), rez_b).T.contiguous()   # (n, ni)
            gi_t = project_interaction(G_t.unsqueeze(-1) * c2.unsqueeze(0), rez_b)       # (m, n, ni)

        best_group_top = None  # grouped interaction: best over phenotypes in the group

        # Per-phenotype regressions in this window
        for y_t, pid in y_iter:
            gxh_slope = gxh_se = gxh_t = gxh_p = None
            cov_int_result = None
            dosage_vals: dict[str, np.ndarray] | None = None
            if dosage_mode:
                K = int(D_resid.shape[2])
                res = run_batch_dosage_regression(
                    y=y_t, D=D_resid, L=L_resid, k_eff=k_eff, device=device
                )
                dof = int(res["dof"])
                b_np = res["betas"].detach().cpu().numpy()                     # (m, K)
                se_np = res["ses"].detach().cpu().numpy()
                t_np = res["tstats"].detach().cpu().numpy()
                cov_np = res["cov"].detach().cpu().numpy().astype(np.float64)   # (m, K, K)
                # Mask ancestries with too few minor alleles in-sample (report-level
                # mask: the design and fstat_joint are unchanged).
                masked = None
                if min_ancestry_ac and min_ancestry_ac > 0 and anc_stats_np is not None:
                    masked = anc_stats_np["ma_count"] < float(min_ancestry_ac)   # (m, K)
                    b_np = np.where(masked, np.nan, b_np)
                    se_np = np.where(masked, np.nan, se_np)
                    t_np = np.where(masked, np.nan, t_np)
                dosage_vals = {}
                for k in range(K):
                    dosage_vals[f"slope_anc{k}"] = b_np[:, k]
                    dosage_vals[f"slope_se_anc{k}"] = se_np[:, k]
                    dosage_vals[f"tstat_anc{k}"] = t_np[:, k]
                    dosage_vals[f"pval_anc{k}"] = get_t_pval(t_np[:, k], dof)
                f_np = res["fstat"].detach().cpu().numpy()
                dosage_vals["fstat_joint"] = f_np
                dosage_vals["pval_joint"] = get_f_pval(f_np, K, dof)
                for i in range(K):
                    for j in range(i + 1, K):
                        var = cov_np[:, i, i] + cov_np[:, j, j] - 2.0 * cov_np[:, i, j]
                        z = (b_np[:, i] - b_np[:, j]) / np.sqrt(np.maximum(var, 1e-30))
                        if masked is not None:
                            z = np.where(masked[:, i] | masked[:, j], np.nan, z)
                        dosage_vals[f"z_diff_anc{i}_anc{j}"] = z
                        dosage_vals[f"pval_diff_anc{i}_anc{j}"] = get_t_pval(z, dof)
                if anc_stats_np is not None:
                    for k in range(K):
                        dosage_vals[f"an_anc{k}"] = anc_stats_np["an"][:, k]
                        dosage_vals[f"ac_anc{k}"] = anc_stats_np["ac"][:, k]
                        dosage_vals[f"af_anc{k}"] = anc_stats_np["af"][:, k]
                        dosage_vals[f"ma_count_anc{k}"] = anc_stats_np["ma_count"][:, k]
                # Lead signal = ancestry with the strongest (unmasked) effect
                abs_t = np.abs(t_np)
                all_nan = np.all(np.isnan(abs_t), axis=1)
                best_k = np.nanargmax(np.where(all_nan[:, None], 0.0, abs_t), axis=1)
                row_ix = np.arange(t_np.shape[0])
                slope = b_np[row_ix, best_k]
                slope_se = se_np[row_ix, best_k]
                tstat = t_np[row_ix, best_k]
                pvals = get_t_pval(tstat, dof)
                perm_max_r2 = np.nan if include_perm else None
            elif interaction_mode:
                pH = int(H_resid.shape[2])
                n_anc = int(n_ancestries) if n_ancestries is not None else pH

                # Joint model: y ~ g + h_0..h_{pH-1} + g*h_0..g*h_{pH-1}
                H_full = torch.cat([H_resid, I_resid], dim=2)  # (m, n, 2*pH)
                betas_all, ses_all, tstats_all = run_batch_regression(
                    y=y_t, G=G_resid, H=H_full, k_eff=k_eff, device=device
                )
                # Columns: 0=g, 1..pH=h_main, pH+1..2*pH=g*h_interaction
                n = int(y_t.shape[0])
                dof = max(n - int(k_eff) - (1 + 2 * pH), 1)

                # Per-ancestry G×H interaction effects
                gxh_slope = np.full((G_resid.shape[0], n_anc), np.nan, dtype=np.float32)
                gxh_se = np.full_like(gxh_slope, np.nan)
                gxh_t = np.full_like(gxh_slope, np.nan)
                gxh_p = np.full_like(gxh_slope, np.nan)

                gxh_betas = betas_all[:, 1 + pH:].detach().cpu().numpy()   # (m, pH)
                gxh_ses = ses_all[:, 1 + pH:].detach().cpu().numpy()
                gxh_tstats = tstats_all[:, 1 + pH:].detach().cpu().numpy()
                for k in range(pH):
                    gxh_slope[:, k] = gxh_betas[:, k]
                    gxh_se[:, k] = gxh_ses[:, k]
                    gxh_t[:, k] = gxh_tstats[:, k]
                    gxh_p[:, k] = get_t_pval(gxh_tstats[:, k], dof)

                # Lead signal = best G×H interaction across ancestries
                best_k = np.nanargmax(np.abs(gxh_t[:, :pH]), axis=1)
                row_ix = np.arange(gxh_t.shape[0])
                slope = gxh_slope[row_ix, best_k]
                slope_se = gxh_se[row_ix, best_k]
                tstat = gxh_t[row_ix, best_k]
                pvals = gxh_p[row_ix, best_k]
                perm_max_r2 = np.nan if include_perm else None
            elif covar_interaction:
                # Joint model: y ~ g + i_1..i_ni + g*i_1..g*i_ni (+ local-ancestry main effects H)
                cov_int_result = run_batch_interaction_regression(
                    y=y_t, G=G_resid, interaction=i_res_t, k_eff=k_eff, device=device,
                    H=H_resid, gi=gi_t,
                )
                int_dof = cov_int_result["dof"]

                # Standard columns: the strongest G×I interaction term per variant
                b_gi = cov_int_result["b_gi"].detach().cpu().numpy().reshape(G_resid.shape[0], ni)
                se_gi = cov_int_result["se_gi"].detach().cpu().numpy().reshape(G_resid.shape[0], ni)
                t_gi = cov_int_result["tstat_gi"].detach().cpu().numpy().reshape(G_resid.shape[0], ni)
                best_k = np.nanargmax(np.where(np.isnan(t_gi), -1.0, np.abs(t_gi)), axis=1)
                row_ix = np.arange(G_resid.shape[0])
                slope = b_gi[row_ix, best_k]
                slope_se = se_gi[row_ix, best_k]
                tstat = t_gi[row_ix, best_k]
                pvals = get_t_pval(tstat, int_dof)
                perm_max_r2 = None
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
            if interaction_mode or dosage_mode or covar_interaction:
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

            beta_np, se_np, tstat_np, pval_np, af_np = (floats[:, i] for i in range(5))
            ma_samples_np, ma_count_np = ints[:, 0], ints[:, 1]

            # Covariate-interaction columns (per term) and the top association
            cov_int_cols: dict[str, np.ndarray] | None = None
            if covar_interaction and cov_int_result is not None:
                int_dof = cov_int_result["dof"]
                cov_int_cols = {}
                for pre in ("g",):
                    b_ = cov_int_result["b_g"].detach().cpu().numpy()
                    s_ = cov_int_result["se_g"].detach().cpu().numpy()
                    t_ = cov_int_result["tstat_g"].detach().cpu().numpy()
                    cov_int_cols.update({"b_g": b_, "se_g": s_, "tstat_g": t_, "pval_g": get_t_pval(t_, int_dof)})
                b_i = cov_int_result["b_i"].detach().cpu().numpy().reshape(n_variants, ni)
                se_i = cov_int_result["se_i"].detach().cpu().numpy().reshape(n_variants, ni)
                t_i = cov_int_result["tstat_i"].detach().cpu().numpy().reshape(n_variants, ni)
                for k in range(ni):
                    suf_i, suf_gi = _interaction_term_columns(interaction_names, k)
                    cov_int_cols[f"b_{suf_i}"] = b_i[:, k]
                    cov_int_cols[f"se_{suf_i}"] = se_i[:, k]
                    cov_int_cols[f"tstat_{suf_i}"] = t_i[:, k]
                    cov_int_cols[f"pval_{suf_i}"] = get_t_pval(t_i[:, k], int_dof)
                    cov_int_cols[f"b_{suf_gi}"] = b_gi[:, k]
                    cov_int_cols[f"se_{suf_gi}"] = se_gi[:, k]
                    cov_int_cols[f"tstat_{suf_gi}"] = t_gi[:, k]
                    cov_int_cols[f"pval_{suf_gi}"] = get_t_pval(t_gi[:, k], int_dof)
                if top_rows is not None:
                    score = np.nanmax(np.abs(t_gi), axis=1)
                    if np.all(np.isnan(score)):
                        ix = 0
                    else:
                        ix = int(np.nanargmax(score))
                    top = {
                        "phenotype_id": str(pid), "variant_id": var_ids[ix],
                        "start_distance": int(var_pos[ix] - ig.phenotype_start[pid]),
                        "end_distance": int(var_pos[ix] - ig.phenotype_end[pid]),
                        "af": float(af_np[ix]), "ma_samples": int(ma_samples_np[ix]),
                        "ma_count": int(ma_count_np[ix]),
                    }
                    for col in covar_int_columns:
                        top[col] = float(cov_int_cols[col][ix])
                    if tests_emt is not None:
                        top["tests_emt"] = int(tests_emt)
                    if group_mode:
                        top["group_id"] = group_id
                        top["num_phenotypes"] = len(id_list)
                        if best_group_top is None or float(score[ix]) > best_group_top[0]:
                            best_group_top = (float(score[ix]), top)
                    else:
                        top_rows.append(top)

            if buffers is not None:
                row_slice = slice(cursor, cursor + n_variants)

                buffers["phenotype_id"][row_slice] = str(pid)
                buffers["variant_id"][row_slice] = var_ids
                np.subtract(var_pos, ig.phenotype_start[pid],
                            out=buffers["start_distance"][row_slice])
                np.subtract(var_pos, ig.phenotype_end[pid],
                            out=buffers["end_distance"][row_slice])

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
                if dosage_mode and dosage_vals is not None:
                    for col in dosage_columns:
                        buffers[col][row_slice] = dosage_vals[col]
                if cov_int_cols is not None:
                    for col in covar_int_columns:
                        buffers[col][row_slice] = cov_int_cols[col]

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

        if best_group_top is not None and top_rows is not None:
            top_rows.append(best_group_top[1])

    if buffers is None:
        return None

    if sink is not None:
        schema = _nominal_parquet_schema(include_perm, ancestry_model, n_ancestries,
                                         covariate_interaction=covar_interaction,
                                         interaction_names=interaction_names)
        table = _buffers_to_arrow(buffers, schema, cursor)
        sink.write(table)
        return None

    if cursor == 0:
        return pd.DataFrame(columns=expected_columns)

    data = {col: buffers[col][:cursor] for col in expected_columns}
    return pd.DataFrame(data, columns=expected_columns)


def _n_ancestries(haplotypes, ancestry_model: str) -> int | None:
    """
    Ancestry count reported in the output.  The interaction model reports one
    block per channel; the dosage model needs every ancestry (``n_pops`` of a
    panel; a single legacy channel means the collapsed two-ancestry case).
    """
    if haplotypes is None:
        return None
    k = int(haplotypes.shape[2])
    if ancestry_model != "dosage":
        return k
    n_pops = getattr(haplotypes, "n_pops", None)
    if n_pops is not None:
        return int(n_pops)
    return 2 if k == 1 else k


def _prepare_interaction_terms(interaction_df, interaction_covariate, covariates_df, sample_ids):
    """
    Normalise ``interaction_df`` (DataFrame samples x terms, or Series) and the
    legacy ``interaction_covariate`` (one term; z-scored) into
    ``(tensor (n, ni) or None, names or None, covariates_to_project)``.
    """
    if interaction_df is not None and interaction_covariate is not None:
        raise ValueError("Pass either interaction_df or interaction_covariate, not both.")
    covariates_use = covariates_df
    if interaction_df is not None:
        if isinstance(interaction_df, pd.Series):
            interaction_df = interaction_df.to_frame(name=interaction_df.name or "i")
        if not isinstance(interaction_df, pd.DataFrame):
            raise ValueError("interaction_df must be a DataFrame (samples x terms) or a Series.")
        idx = pd.Index(interaction_df.index.astype(str))
        want = pd.Index(pd.Index(sample_ids).astype(str))
        missing = want.difference(idx)
        if len(missing):
            raise ValueError(f"interaction_df is missing {len(missing)} samples (first: {list(missing[:5])})")
        vals = interaction_df.set_index(idx).loc[want].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("interaction_df contains non-finite values.")
        if (vals.std(axis=0) <= 0).any():
            raise ValueError("interaction_df has a constant column.")
        names = [str(c) for c in interaction_df.columns]
        if len(set(names)) != len(names):
            raise ValueError("interaction_df column names must be unique.")
        reserved = {"g", "i", "gi"} | set(_covariate_interaction_columns())
        bad = [n for n in names if n in reserved]
        if bad and len(names) > 1:
            raise ValueError(f"interaction_df column names {bad} collide with output columns; rename them.")
        # a term also present among the covariates is projected as an interaction, not a covariate
        if covariates_df is not None:
            drop = [c for c in covariates_df.columns if str(c) in names]
            if drop:
                covariates_use = covariates_df.drop(columns=drop)
        return vals.astype(np.float32), names, covariates_use
    if interaction_covariate is not None:
        if isinstance(interaction_covariate, str) and covariates_df is not None \
                and interaction_covariate in covariates_df.columns:
            covariates_use = covariates_df.drop(columns=[interaction_covariate])
        vec = prepare_interaction_covariate(interaction_covariate, covariates_df, sample_ids)
        name = interaction_covariate if isinstance(interaction_covariate, str) else \
            (getattr(interaction_covariate, "name", None) or "i")
        return vec.reshape(-1, 1), [str(name)], covariates_use
    return None, None, covariates_use


def _paired_covariate_context(paired_covariate_df, covariates_df, phenotype_df_resid_index, phenotype_df_raw,
                              device, tensorqtl_flavor):
    """
    Phenotype-specific covariates (tensorQTL ``paired_covariate_df``,
    phenotypes x samples): one Residualizer per phenotype that includes the
    shared covariates plus that phenotype's extra column, and the phenotype
    residualized with it.
    """
    if paired_covariate_df is None:
        return None
    if not paired_covariate_df.columns.equals(phenotype_df_raw.columns):
        raise ValueError("paired_covariate_df must be phenotypes x samples with the phenotype sample order.")
    unknown = paired_covariate_df.index.difference(phenotype_df_raw.index)
    if len(unknown):
        raise ValueError(f"paired_covariate_df has {len(unknown)} phenotypes not in phenotype_df (first: {list(unknown[:5])})")
    ctx = {}
    C = covariates_df.values if covariates_df is not None else np.empty((phenotype_df_raw.shape[1], 0))
    for pid in paired_covariate_df.index:
        if pid not in phenotype_df_resid_index:
            continue
        Ci = np.c_[C, paired_covariate_df.loc[pid].to_numpy(dtype=np.float64)]
        rez_i = Residualizer(torch.tensor(Ci, dtype=torch.float32, device=device), tensorqtl_flavor=tensorqtl_flavor)
        y = torch.tensor(phenotype_df_raw.loc[pid].to_numpy(dtype=np.float32), device=device).unsqueeze(0)
        ctx[pid] = (rez_i, rez_i.transform(y, center=True)[0].squeeze(0))
    return ctx


def _finalize_top_assoc(top_rows, interaction_names, run_eigenmt, group_mode) -> pd.DataFrame:
    """Top interaction association per phenotype (group) with optional eigenMT correction."""
    if not top_rows:
        return pd.DataFrame()
    top_df = pd.DataFrame(top_rows).set_index("phenotype_id")
    if run_eigenmt and "tests_emt" in top_df.columns and (interaction_names is None or len(interaction_names) == 1):
        from ..eigenmt import padjust_bh
        pval_gi = top_df["pval_gi"].to_numpy(dtype=np.float64)
        scale = top_df["tests_emt"].to_numpy(dtype=np.float64)
        if group_mode and "num_phenotypes" in top_df.columns:
            scale = scale * top_df["num_phenotypes"].to_numpy(dtype=np.float64)
        top_df["pval_emt"] = np.minimum(scale * pval_gi, 1.0)
        ok = np.isfinite(top_df["pval_emt"].to_numpy())
        adj = np.full(len(top_df), np.nan)
        if ok.any():
            adj[ok] = padjust_bh(top_df["pval_emt"].to_numpy()[ok])
        top_df["pval_adj_bh"] = adj
    return top_df


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
        min_ancestry_ac: float = 0.0,
        interaction_df: Optional[object] = None,
        maf_threshold_interaction: float = 0.0,
        paired_covariate_df: Optional[pd.DataFrame] = None,
        write_top: bool = True,
        write_stats: bool = True,
        run_eigenmt: bool = False,
        top_assoc_path: Optional[str] = None,
    ) -> pd.DataFrame:
    """
    Nominal cis-QTL scan with optional permutations and local ancestry.

    Adjusts for covariates by residualizing y, G, and H across samples using the
    same Residualizer (projection onto the orthogonal complement of C). Without
    covariates every matrix is centred, which is equivalent to an intercept.

    Parameters
    ----------
    preload_haplotypes : bool, default True
        When haplotypes are provided, pre-load them into a contiguous torch.Tensor
        on the requested device to avoid per-batch host<->device transfers.
    ancestry_model : {"main", "interaction", "dosage"}
        ``"main"`` adjusts for local-ancestry counts; ``"interaction"`` fits GxH
        terms; ``"dosage"`` fits one effect per ancestry and also reports the
        per-ancestry in-sample allele statistics ``an_anc{k}``, ``ac_anc{k}``,
        ``af_anc{k}``, ``ma_count_anc{k}``.
    min_ancestry_ac : float, default 0
        Dosage model only. Ancestries whose in-sample minor-allele count
        (``ma_count_anc{k}``) is below this value get NaN effect estimates and
        NaN pairwise differences; the variant row is kept and ``fstat_joint``
        is unchanged.
    interaction_covariate : str, Series or array, optional
        One genotype-by-covariate interaction term (z-scored).
    interaction_df : DataFrame (samples x terms) or Series, optional
        tensorQTL-style interaction term(s), used as given (not scaled); fits
        ``y ~ g + i_1..i_k + g:i_1..g:i_k`` (+ local-ancestry main effects with
        a panel).  With one term the output columns are ``b_g/se_g/tstat_g/
        pval_g``, ``b_i/...``, ``b_gi/...``; with several terms the variable
        names are substituted (``b_<name>``, ``b_g-<name>``, ...).  The
        ``slope``/``pval_nominal`` columns report the strongest ``g:i`` term.
    maf_threshold_interaction : float, default 0
        With exactly one interaction term, keep variants with MAF >= this value
        in both the lower and upper half of the interaction variable
        (tensorQTL default 0.05; the CLI uses that default).
    paired_covariate_df : DataFrame (phenotypes x samples), optional
        One phenotype-specific covariate (e.g. a cell-type proportion or
        a local PEER factor); phenotypes absent from it use the shared
        covariates only.  Ungrouped mapping only.
    write_top, write_stats : bool
        Interaction scans only: write the top association per phenotype to
        ``top_assoc_path`` (default ``<out_dir>/<out_prefix>.top_assoc.txt.gz``)
        and the pair-level Parquet files.  ``write_stats=False`` (tensorQTL's
        ``--best_only``) skips the pair-level output.
    run_eigenmt : bool
        Interaction scans only: add ``tests_emt`` (eigenMT effective number of
        tests) to the top associations and, with one term, ``pval_emt`` and
        ``pval_adj_bh``.

    Returns
    -------
    ``out_dir=None``: the pair-level DataFrame (interaction scans also carry
    ``.attrs["top_assoc"]``), or the top-association DataFrame when
    ``write_stats=False``.  With ``out_dir``: ``None`` (files written), or the
    top-association DataFrame when ``write_top=False``.
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)
    logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
    sync = (torch.cuda.synchronize if device == "cuda" else None)

    # Interaction term(s)
    interaction_arr, interaction_names, covariates_use = _prepare_interaction_terms(
        interaction_df, interaction_covariate, covariates_df, phenotype_df.columns)
    has_interaction = interaction_arr is not None
    ni = 0 if interaction_arr is None else int(interaction_arr.shape[1])

    # Header (tensorQTL-style)
    logger.write("cis-QTL mapping: nominal associations for all variant–phenotype pairs")
    logger.write(f"  * device: {device}")
    logger.write(f"  * {phenotype_df.shape[1]} samples")
    logger.write(f"  * {phenotype_df.shape[0]} phenotypes")
    logger.write(f"  * {variant_df.shape[0]} variants")
    logger.write(f"  * cis-window: ±{window:,}")
    if maf_threshold and maf_threshold > 0:
        logger.write(f"  * applying in-sample {maf_threshold:g} MAF filter")
    if covariates_use is not None:
        logger.write(f"  * {covariates_use.shape[1]} covariates")
    if paired_covariate_df is not None:
        logger.write("  * including phenotype-specific covariate")
    if haplotypes is not None:
        K = int(haplotypes.shape[2])
        logger.write(f"  * including local ancestry channels (K={K})")
    if ancestry_model == "interaction":
        logger.write("  * ancestry model: interaction (GxH)")
    elif ancestry_model == "dosage":
        logger.write("  * ancestry model: ancestry-specific dosage "
                     f"({'posterior-expected' if getattr(haplotypes, 'use_posterior', False) else 'hard-call'} ancestry counts)")
    if has_interaction:
        logger.write(f"  * including {ni} interaction term(s): {', '.join(interaction_names)}"
                     + (" (+ local ancestry main effects)" if haplotypes is not None else ""))
        if maf_threshold_interaction and maf_threshold_interaction > 0:
            if ni == 1:
                logger.write(f"    * using {maf_threshold_interaction:g} MAF threshold (both halves)")
            else:
                logger.write("    * maf_threshold_interaction ignored with >1 interaction terms")
        if run_eigenmt:
            logger.write("    * computing eigenMT correction for the top associations")
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

    # Validate interaction / dosage arguments
    validate_ancestry_model(
        ancestry_model, haplotypes=haplotypes,
        interaction_covariate=interaction_arr if has_interaction else None,
        allow_dosage=True, caller="map_nominal",
    )
    if ancestry_model == "dosage" and nperm is not None and nperm > 0:
        raise ValueError("ancestry_model='dosage' does not support nperm (permutations) yet.")
    if min_ancestry_ac and min_ancestry_ac > 0:
        if ancestry_model != "dosage":
            raise ValueError("min_ancestry_ac is only meaningful with ancestry_model='dosage'.")
        logger.write(f"  * masking ancestry effects with in-sample minor-allele count < {min_ancestry_ac:g}")
    if paired_covariate_df is not None and group_s is not None:
        raise ValueError("paired_covariate_df is not supported with phenotype groups.")
    if has_interaction and nperm is not None and nperm > 0:
        raise ValueError("interaction terms do not support nperm (permutations).")
    if (not write_stats or run_eigenmt) and not has_interaction:
        raise ValueError("write_stats=False / run_eigenmt require an interaction term.")

    interaction_cov_t = None
    interaction_mask_t = None
    if has_interaction:
        interaction_cov_t = torch.as_tensor(interaction_arr, dtype=torch.float32, device=device)
        if ni == 1 and maf_threshold_interaction and maf_threshold_interaction > 0:
            interaction_mask_t = interaction_half_mask(interaction_arr[:, 0]).to(device)
        else:
            maf_threshold_interaction = 0.0

    # Residualize once using the filtered phenotypes from the generator
    Y = torch.tensor(ig.phenotype_df.values, dtype=torch.float32, device=device)
    with logger.time_block(" Residualizing phenotypes", sync=sync):
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_use,
                                                          device, tensorqtl_flavor)
    paired_ctx = _paired_covariate_context(paired_covariate_df, covariates_use, ig.phenotype_df.index,
                                           phenotype_df, device, tensorqtl_flavor)

    ig.phenotype_df = pd.DataFrame(
        Y_resid.cpu().numpy(), index=ig.phenotype_df.index, columns=ig.phenotype_df.columns
    )

    # Pre-compute per-chromosome phenotype counts for logging
    phenotype_counts = ig.phenotype_pos_df['chr'].value_counts().to_dict()
    n_anc = _n_ancestries(haplotypes, ancestry_model)
    top_rows: list | None = [] if has_interaction else None
    group_mode = group_s is not None
    core_kwargs = dict(
        maf_threshold=maf_threshold, logger=logger, ancestry_model=ancestry_model,
        n_ancestries=n_anc, interaction_covariate_t=interaction_cov_t,
        min_ancestry_ac=min_ancestry_ac, interaction_names=interaction_names,
        interaction_mask_t=interaction_mask_t, maf_threshold_interaction=maf_threshold_interaction,
        paired_ctx=paired_ctx, top_rows=top_rows, run_eigenmt=run_eigenmt,
        collect_pairs=write_stats,
    )

    def _top_df():
        return _finalize_top_assoc(top_rows, interaction_names, run_eigenmt, group_mode)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        include_perm = nperm is not None and nperm > 0
        schema = _nominal_parquet_schema(include_perm, ancestry_model, n_anc,
                                         covariate_interaction=has_interaction,
                                         interaction_names=interaction_names)
        with logger.time_block("Nominal scan (per-chrom streaming)", sync=sync, sec=False):
            for chrom in ig.chrs:
                out_path = os.path.join(out_dir, f"{out_prefix}.{chrom}.parquet")
                chrom_total = int(phenotype_counts.get(chrom, 0))
                if logger.verbose:
                    logger.write(f"    Mapping chromosome {chrom} ({chrom_total} phenotypes)")
                chrom_start = time.time()
                with logger.time_block(f"{chrom}: map_nominal", sync=sync):
                    if write_stats:
                        target_rows = _count_cis_pairs(ig, chrom)
                        with AsyncParquetSink(
                                out_path, schema=schema, compression=compression,
                                row_group_size=DEFAULT_ROW_GROUP_SIZE,
                                use_dictionary=("phenotype_id",), write_statistics=False,
                                column_order=list(schema.names), max_queue_items=4,
                        ) as sink:
                            _run_nominal_core(
                                ig, variant_df, rez, nperm, device, chrom=chrom, sink=sink,
                                target_rows=target_rows, total_phenotypes=chrom_total, **core_kwargs,
                            )
                    else:
                        _run_nominal_core(
                            ig, variant_df, rez, nperm, device, chrom=chrom, sink=None,
                            total_phenotypes=chrom_total, **core_kwargs,
                        )
                if logger.verbose:
                    elapsed = time.time() - chrom_start
                    logger.write(f"    Chromosome {chrom} completed in {elapsed:.2f}s")
        if has_interaction:
            top_df = _top_df()
            if write_top:
                path = top_assoc_path or os.path.join(out_dir, f"{out_prefix}.top_assoc.txt.gz")
                top_df.to_csv(path, sep="\t", float_format="%.6g")
                logger.write(f"  * wrote top associations to {path}")
            else:
                return top_df
        return None if not return_df else pd.DataFrame([])

    total_phenotypes = int(ig.phenotype_df.shape[0])
    if logger.verbose:
        logger.write(f"    Mapping all chromosomes ({total_phenotypes} phenotypes)")
    overall_start = time.time()
    with logger.time_block("Computing associations (nominal)", sync=sync):
        results = _run_nominal_core(ig, variant_df, rez, nperm, device,
                                    chrom=None, sink=None, total_phenotypes=total_phenotypes,
                                    **core_kwargs)
    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed nominal scan in {elapsed / 60:.2f} min")
    if has_interaction:
        top_df = _top_df()
        if results is None:
            return top_df
        results.attrs["top_assoc"] = top_df
    return results


def map_nominal_loci(
    locus_df: pd.DataFrame,
    genotype_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    covariates_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run nominal mapping only for pre-specified loci.

    This is a thin wrapper around :func:`map_nominal` that filters phenotypes
    and adjusts windows to match specified loci.

    Parameters
    ----------
    locus_df : DataFrame
        Must contain ``phenotype_id`` and ``chr`` plus either
        ``start``/``end`` (explicit window, tested as-is) or ``position``
        (tested as ``position ± window``). Several loci per phenotype are
        allowed; the phenotype is then reported once per locus with the
        suffix ``__locus{i}`` appended to ``phenotype_id``.
    genotype_df, variant_df, phenotype_df, covariates_df
        As in :func:`map_nominal`.
    window : int, default 1_000_000
        Padding applied to ``position``; ignored when ``start``/``end`` are
        given. The padding is applied exactly once.
    **kwargs
        Passed through to :func:`map_nominal` (``phenotype_pos_df`` and
        ``window`` are managed here and may not be passed).

    Returns
    -------
    DataFrame
        Nominal scan results for the specified loci only (``out_dir`` defaults
        to ``None`` here so a DataFrame is returned).
    """
    if "phenotype_pos_df" in kwargs:
        raise ValueError("map_nominal_loci builds phenotype_pos_df from locus_df; do not pass it.")
    window = int(kwargs.pop("window", 1_000_000))
    kwargs.setdefault("out_dir", None)
    logger = kwargs.get("logger") or SimpleLogger(verbose=kwargs.get("verbose", True),
                                                   timestamps=True)
    kwargs["logger"] = logger

    required = {"phenotype_id", "chr"}
    missing = required - set(locus_df.columns)
    if missing:
        raise ValueError(f"locus_df is missing required columns: {sorted(missing)}")
    locus_df = locus_df.copy()
    if "position" in locus_df.columns:
        pos = locus_df["position"].astype(np.int64)
        locus_df["start"] = np.maximum(pos - window, 1)
        locus_df["end"] = pos + window
    elif not {"start", "end"}.issubset(locus_df.columns):
        raise ValueError("locus_df needs either 'position' or both 'start' and 'end'.")
    locus_df["start"] = np.maximum(locus_df["start"].astype(np.int64), 1)
    locus_df["end"] = locus_df["end"].astype(np.int64)

    known = locus_df["phenotype_id"].isin(phenotype_df.index)
    if (~known).any():
        dropped = locus_df.loc[~known, "phenotype_id"].unique()
        logger.write(f"  * dropping {len(dropped)} loci whose phenotype is not in phenotype_df")
        locus_df = locus_df.loc[known]
    if locus_df.empty:
        raise ValueError("No loci left after matching phenotype_df.")

    # One output phenotype per locus; suffix only when a phenotype has >1 locus.
    counts = locus_df.groupby("phenotype_id")["phenotype_id"].transform("size")
    rank = locus_df.groupby("phenotype_id").cumcount()
    out_ids = np.where(counts > 1,
                       locus_df["phenotype_id"].astype(str) + "__locus" + rank.astype(str),
                       locus_df["phenotype_id"].astype(str))
    phenotype_sub = phenotype_df.loc[locus_df["phenotype_id"].to_numpy()].copy()
    phenotype_sub.index = pd.Index(out_ids, name=phenotype_df.index.name)
    pos_df = pd.DataFrame({
        "chr": locus_df["chr"].astype(str).to_numpy(),
        "start": locus_df["start"].to_numpy(),
        "end": locus_df["end"].to_numpy(),
    }, index=phenotype_sub.index)

    # window=0: the generator uses exactly [start, end] (already padded above).
    return map_nominal(
        genotype_df=genotype_df,
        variant_df=variant_df,
        phenotype_df=phenotype_sub,
        phenotype_pos_df=pos_df,
        covariates_df=covariates_df,
        window=0,
        **kwargs,
    )
