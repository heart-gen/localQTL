import torch, time
import numpy as np
import pandas as pd
from typing import Optional

try:
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except Exception:
    pass

from ..utils import SimpleLogger, subseed
from .._torch_utils import to_device_tensor
from ..stats import beta_approx_pval, get_t_pval
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, filter_by_maf
from ..regression_kernels import (
    Residualizer,
    run_batch_regression_with_permutations
)
from ._permute import make_perm_ix, roll_for_key, compute_perm_r2_max
from ._buffer import (
    allocate_result_buffers,
    ensure_capacity, write_row,
    buffers_to_dataframe
)
from .common import (
    align_like_casefold,
    residualize_batch,
    dosage_vector_for_covariate
)

__all__ = [
    "map_independent",
]

def _auto_perm_chunk(n_variants: int, nperm: int, pH: int = 0, safety: float = 0.5) -> int:
    if not torch.cuda.is_available() or n_variants <= 0:
        return min(nperm, 2048)
    free, _ = torch.cuda.mem_get_info()
    bytes_per_col = 4 * n_variants * (3 + 2 * max(pH, 0))
    if bytes_per_col <= 0:
        return min(nperm, 2048)
    max_chunk = int((free * safety) // bytes_per_col)
    p2 = 1 << (max_chunk.bit_length() - 1) # round down to power of two
    return max(1, min(nperm, p2))


def _nanmax(x: torch.Tensor, dim: int):
    if hasattr(torch, "nanmax"):
        return torch.nanmax(x, dim=dim)
    replace = torch.full_like(x, float("-inf"))
    return torch.max(torch.where(torch.isnan(x), replace, x), dim=dim)

def _run_independent_core(
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        signif_seed_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True, seed: int | None = None,
        chrom: str | None = None, perm_ix_t: torch.Tensor | None = None,
        perm_chunk: int = 2048, logger: SimpleLogger | None = None,
        total_items: int | None = None, item_label: str = "phenotypes",
        tensorqtl_flavor: bool = False,
) -> pd.DataFrame:
    """Forward–backward independent mapping for ungrouped phenotypes."""
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance", "end_distance",
        "num_var", "slope", "slope_se", "tstat", "r2_nominal", "pval_nominal",
        "pval_perm", "pval_beta", "beta_shape1", "beta_shape2", "ma_samples",
        "ma_count", "af", "true_dof", "pval_true_dof", "rank",
    ]
    dtype_map = {
        "phenotype_id": object,
        "variant_id": object,
        "start_distance": np.int32,
        "end_distance": np.int32,
        "num_var": np.int32,
        "slope": np.float32,
        "slope_se": np.float32,
        "tstat": np.float32,
        "r2_nominal": np.float32,
        "pval_nominal": np.float32,
        "pval_perm": np.float32,
        "pval_beta": np.float32,
        "beta_shape1": np.float32,
        "beta_shape2": np.float32,
        "ma_samples": np.int32,
        "ma_count": np.float32,
        "af": np.float32,
        "true_dof": np.int32,
        "pval_true_dof": np.float32,
        "rank": np.int32,
    }
    buffers = allocate_result_buffers(expected_columns, dtype_map, signif_seed_df.shape[0])
    cursor = 0
    processed = 0

    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, int(total_items) // 10) if total_items else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"

    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int32)

    # Basic alignment checks
    covariates_base_t: torch.Tensor | None = None
    if covariates_df is not None:
        covariates_df = align_like_casefold(
            covariates_df,
            ig.phenotype_df.columns,
            axis="index",
            what="sample IDs in covariates_df.index",
            strict=True,
        )
        covariates_base_t = to_device_tensor(
            covariates_df.to_numpy(np.float32, copy=False), device,
            dtype=torch.float32
        )

    var_in_frame = set(variant_df.index)

    for batch in ig.generate_data(chrom=chrom):
        if len(batch) == 4:
            p, G_block, v_idx, pid = batch
            H_block = None
        elif len(batch) == 5 and not isinstance(batch[3], (list, tuple)):
            p, G_block, v_idx, H_block, pid = batch
        else:
            raise ValueError("Unexpected batch shape in _run_independent_core (ungrouped).")

        pid = str(pid)
        if pid not in signif_seed_df.index:
            # phenotype not FDR-significant -> skip
            continue
        seed_row = signif_seed_df.loc[pid]

        # Tensors for the window
        y_t = to_device_tensor(p, device, dtype=torch.float32)
        G_t = to_device_tensor(G_block, device, dtype=torch.float32)
        if H_block is None:
            H_t = None
        else:
            H_t = to_device_tensor(H_block, device, dtype=torch.float32)

        # Impute & filter (and optional MAF)
        G_imputed, keep_mask, _ = impute_mean_and_filter(G_t)
        if G_imputed.shape[0] == 0:
            continue

        mask_cpu = keep_mask.detach().cpu().numpy()
        v_idx = v_idx[mask_cpu]
        if H_t is not None:
            H_t = H_t[mask_cpu]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1]  # drop one ancestry channel
            H_t = H_t.contiguous()

        G_t = G_imputed.contiguous()

        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            if keep_maf.sum().item() == 0:
                continue
            mask_cpu = keep_maf.detach().cpu().numpy()
            G_t = G_t[keep_maf].contiguous()
            v_idx = v_idx[mask_cpu]
            if H_t is not None:
                H_t = H_t[keep_maf].contiguous()

        pH = 0 if H_t is None else int(H_t.shape[2])
        perm_chunk_local = _auto_perm_chunk(G_t.shape[0], nperm, pH=pH)
        if perm_chunk is not None and perm_chunk > 0:
            perm_chunk_local = min(perm_chunk_local, int(perm_chunk))

        # Build per-phenotype generator
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(subseed(seed, pid))

        # Permutation indices for this phenotype
        n = y_t.shape[0]
        if perm_ix_t is not None:
            perm_ix_pid = roll_for_key(perm_ix_t, pid, seed)
        else:
            local_seed = subseed(seed, pid) if seed is not None else None
            perm_ix_pid = make_perm_ix(n, nperm, device, local_seed)

        # Forward pass
        forward_records: list[dict[str, object]] = []
        base_record = {col: seed_row.get(col) for col in expected_columns if col != "rank"}
        forward_records.append(base_record)
        dosage_dict: dict[str, torch.Tensor] = {}
        seed_vid = str(seed_row["variant_id"])
        if seed_vid in var_in_frame and seed_vid in ig.genotype_df.index:
            dosage_dict[seed_vid] = torch.as_tensor(
                dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=seed_vid,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                ),
                dtype=torch.float32,
                device=device,
            )

        while True:
            extras = [dosage_dict[v] for v in dosage_dict]
            rez_aug = None
            with torch.no_grad():
                if covariates_base_t is not None or extras:
                    components = []
                    if covariates_base_t is not None:
                        components.append(covariates_base_t)
                    if extras:
                        components.append(torch.stack(extras, dim=1))
                    C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                    rez_aug = Residualizer(C_aug_t, tensorqtl_flavor=tensorqtl_flavor)
                y_resid, G_resid, H_resid = residualize_batch(
                    y_t, G_t, H_t, rez_aug, center=True, group=False
                )
                k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                betas, ses, tstats, r2_perm = compute_perm_r2_max(
                    y_resid=y_resid,
                    G_resid=G_resid,
                    H_resid=H_resid,
                    k_eff=k_eff,
                    perm_ix=perm_ix_pid,
                    device=device,
                    perm_chunk=perm_chunk_local,
                    return_nominal=True,
                )

                p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                n = int(y_resid.shape[0])
                dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                t_g = tstats[:, 0].double()
                t2 = t_g.pow(2)
                r2_nominal_vec = (t2 / (t2 + float(dof))).to(torch.float32)
                r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                ix = int(ix_t.item())
                if random_tiebreak:
                    ties = torch.nonzero(
                        torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12),
                        as_tuple=True,
                    )[0]
                    if ties.numel() > 1:
                        if gen is None:
                            choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                        else:
                            choice_tensor = torch.randint(
                                0, ties.numel(), (1,), device=ties.device, generator=gen
                            )
                        ix = int(ties[int(choice_tensor.item())].item())
                        r2_max_t = r2_nominal_vec[ix]
                beta = float(betas[ix, 0].item())
                se = float(ses[ix, 0].item())
                tval = float(t_g[ix].item())
                r2_nom = float(r2_max_t.item())
                pval_perm = (
                    (r2_perm >= r2_max_t).sum().add_(1).float() / (r2_perm.numel() + 1)
                ).item()
                r2_perm_np = r2_perm.detach().cpu().numpy()
                if beta_approx:
                    pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                        r2_perm_np, r2_nom, dof_init=dof
                    )
                else:
                    pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
                stop_pval = float(pval_beta)
                if not np.isfinite(stop_pval):
                    stop_pval = pval_perm
                pval_nominal = float(get_t_pval(tval, dof))
                num_var = int(G_resid.shape[0])
                g_sel = G_t[ix].contiguous()
                s = g_sel.sum()
                n2 = 2.0 * g_sel.numel()
                af = (s / n2).item()
                gt_half = (g_sel > 0.5)
                sum_gt_half = g_sel[gt_half].sum()
                if af <= 0.5:
                    ma_samples = int(gt_half.sum().item())
                    ma_count = float(sum_gt_half.item())
                else:
                    ma_samples = int((g_sel < 1.5).sum().item())
                    ma_count = float(n2 - sum_gt_half.item())

            if stop_pval > signif_threshold:
                break

            var_id = idx_to_id[v_idx[ix]]
            var_pos = int(pos_arr[v_idx[ix]])
            start_pos = ig.phenotype_start[pid]
            end_pos = ig.phenotype_end[pid]
            start_distance = int(var_pos - start_pos)
            end_distance = int(var_pos - end_pos)
            forward_records.append({
                "phenotype_id": pid,
                "variant_id": var_id,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "num_var": num_var,
                "slope": beta,
                "slope_se": se,
                "tstat": tval,
                "r2_nominal": r2_nom,
                "pval_nominal": pval_nominal,
                "pval_perm": pval_perm,
                "pval_beta": float(pval_beta),
                "beta_shape1": float(a_hat),
                "beta_shape2": float(b_hat),
                "af": af,
                "ma_samples": ma_samples,
                "ma_count": ma_count,
                "true_dof": int(dof),
                "pval_true_dof": p_true,
            })

            if var_id in var_in_frame and var_id in ig.genotype_df.index and var_id not in dosage_dict:
                dosage_dict[var_id] = torch.as_tensor(
                    dosage_vector_for_covariate(
                        genotype_df=ig.genotype_df,
                        variant_id=var_id,
                        sample_order=ig.phenotype_df.columns,
                        missing=missing,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

        if not forward_records:
            continue

        if len(forward_records) > 1:
            kept_records: list[dict[str, object]] = []
            selected = [rec["variant_id"] for rec in forward_records]
            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]
                extras = [dosage_dict[v] for v in kept]
                rez_aug = None
                with torch.no_grad():
                    if covariates_base_t is not None or extras:
                        components = []
                        if covariates_base_t is not None:
                            components.append(covariates_base_t)
                        if extras:
                            components.append(torch.stack(extras, dim=1))
                        C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                        rez_aug = Residualizer(C_aug_t, tensorqtl_flavor=tensorqtl_flavor)
                    y_resid, G_resid, H_resid = residualize_batch(
                        y_t, G_t, H_t, rez_aug, center=True, group=False
                    )
                    k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                    betas, ses, tstats, r2_perm = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=G_resid,
                        H_resid=H_resid,
                        k_eff=k_eff,
                        perm_ix=perm_ix_pid,
                        device=device,
                        perm_chunk=perm_chunk_local,
                        return_nominal=True,
                    )
                    p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                    dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                    t_g = tstats[:, 0].double()
                    t2 = t_g.pow(2)
                    r2_nominal_vec = (t2 / (t2 + float(dof))).to(torch.float32)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    if random_tiebreak:
                        ties = torch.nonzero(
                            torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12),
                            as_tuple=True,
                        )[0]
                        if ties.numel() > 1:
                            if gen is None:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                            else:
                                choice_tensor = torch.randint(
                                    0, ties.numel(), (1,), device=ties.device, generator=gen
                                )
                            ix = int(ties[int(choice_tensor.item())].item())
                            r2_max_t = r2_nominal_vec[ix]
                    beta = float(betas[ix, 0].item())
                    se = float(ses[ix, 0].item())
                    tval = float(t_g[ix].item())
                    r2_nom = float(r2_max_t.item())
                    pval_perm = (
                        (r2_perm >= r2_max_t).sum().add_(1).float() / (r2_perm.numel() + 1)
                    ).item()
                    r2_perm_np = r2_perm.detach().cpu().numpy()
                    if beta_approx:
                        pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                            r2_perm_np, r2_nom, dof_init=dof
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
                    stop_pval = float(pval_beta)
                    if not np.isfinite(stop_pval):
                        stop_pval = pval_perm
                    pval_nominal = float(get_t_pval(tval, dof))
                    num_var = int(G_resid.shape[0])
                    g_sel = G_t[ix].contiguous()
                    s = g_sel.sum()
                    n2 = 2.0 * g_sel.numel()
                    af = (s / n2).item()
                    gt_half = (g_sel > 0.5)
                    sum_gt_half = g_sel[gt_half].sum()
                    if af <= 0.5:
                        ma_samples = int(gt_half.sum().item())
                        ma_count = float(sum_gt_half.item())
                    else:
                        ma_samples = int((g_sel < 1.5).sum().item())
                        ma_count = float(n2 - sum_gt_half.item())

                if stop_pval <= signif_threshold:
                    var_id = idx_to_id[v_idx[ix]]
                    var_pos = int(pos_arr[v_idx[ix]])
                    start_pos = ig.phenotype_start[pid]
                    end_pos = ig.phenotype_end[pid]
                    kept_records.append({
                        "phenotype_id": pid,
                        "variant_id": var_id,
                        "start_distance": int(var_pos - start_pos),
                        "end_distance": int(var_pos - end_pos),
                        "num_var": num_var,
                        "slope": beta,
                        "slope_se": se,
                        "tstat": tval,
                        "r2_nominal": r2_nom,
                        "pval_nominal": pval_nominal,
                        "pval_perm": pval_perm,
                        "pval_beta": float(pval_beta),
                        "beta_shape1": float(a_hat),
                        "beta_shape2": float(b_hat),
                        "af": af,
                        "ma_samples": ma_samples,
                        "ma_count": ma_count,
                        "true_dof": int(dof),
                        "pval_true_dof": p_true,
                        "rank": int(rk),
                    })

            if not kept_records:
                continue
            records_to_write = kept_records
        else:
            records_to_write = forward_records

        for rk, rec in enumerate(records_to_write, start=1):
            rec["rank"] = rk
            buffers = ensure_capacity(buffers, cursor, 1)
            write_row(buffers, cursor, rec)
            cursor += 1

        processed += 1
        if (
            logger.verbose
            and total_items
            and progress_interval
            and (processed % progress_interval == 0 or processed == total_items)
        ):
            logger.write(
                f"      processed {processed}/{total_items} {item_label} on {chrom_label}"
            )

    return buffers_to_dataframe(expected_columns, buffers, cursor)




def _run_independent_core_group(
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        seed_by_group_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True, seed: int | None = None,
        chrom: str | None = None, perm_ix_t: torch.Tensor | None = None,
        perm_chunk: int = 2048, logger: SimpleLogger | None = None,
        total_items: int | None = None, item_label: str = "phenotype groups",
        tensorqtl_flavor: bool = False,
) -> pd.DataFrame:
    """Forward-backward independent mapping for grouped phenotypes."""
    expected_columns = [
        "group_id", "group_size", "phenotype_id", "variant_id", "start_distance",
        "end_distance", "num_var", "slope", "slope_se", "tstat", "r2_nominal",
        "pval_nominal", "pval_perm", "pval_beta", "beta_shape1", "beta_shape2",
        "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof", "rank",
    ]
    dtype_map = {
        "group_id": object,
        "group_size": np.int32,
        "phenotype_id": object,
        "variant_id": object,
        "start_distance": np.int32,
        "end_distance": np.int32,
        "num_var": np.int32,
        "slope": np.float32,
        "slope_se": np.float32,
        "tstat": np.float32,
        "r2_nominal": np.float32,
        "pval_nominal": np.float32,
        "pval_perm": np.float32,
        "pval_beta": np.float32,
        "beta_shape1": np.float32,
        "beta_shape2": np.float32,
        "ma_samples": np.int32,
        "ma_count": np.float32,
        "af": np.float32,
        "true_dof": np.int32,
        "pval_true_dof": np.float32,
        "rank": np.int32,
    }
    buffers = allocate_result_buffers(expected_columns, dtype_map, seed_by_group_df.shape[0])
    cursor = 0
    processed = 0

    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, int(total_items) // 10) if total_items else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"

    var_in_frame = set(variant_df.index)
    geno_has_variant = set(ig.genotype_df.index)
    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int64, copy=False)

    covariates_base_t: torch.Tensor | None = None
    if covariates_df is not None:
        covariates_df = align_like_casefold(
            covariates_df,
            ig.phenotype_df.columns,
            axis="index",
            what="sample IDs in covariates_df.index",
            strict=True,
        )
        covariates_base_t = to_device_tensor(
            covariates_df.to_numpy(np.float32, copy=False), device,
            dtype=torch.float32
        )

    for batch in ig.generate_data(chrom=chrom):
        if len(batch) == 5:
            _, G_block, v_idx, ids, group_id = batch
            H_block = None
        elif len(batch) == 6:
            _, G_block, v_idx, H_block, ids, group_id = batch
        else:
            raise ValueError("Unexpected grouped batch shape in _run_independent_core_group.")

        seed_rows = seed_by_group_df[seed_by_group_df["group_id"] == group_id]
        if seed_rows.empty:
            continue
        seed_row = seed_rows.iloc[0]
        seed_vid = str(seed_row["variant_id"])

        G_t = to_device_tensor(G_block, device, dtype=torch.float32)
        if H_block is None:
            H_t = None
        else:
            H_t = to_device_tensor(H_block, device, dtype=torch.float32)

        G_imputed, keep_mask, _ = impute_mean_and_filter(G_t)
        if G_imputed.shape[0] == 0:
            continue

        mask_cpu = keep_mask.detach().cpu().numpy()
        v_idx = v_idx[mask_cpu]
        if H_t is not None:
            H_t = H_t[mask_cpu]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1]
            H_t = H_t.contiguous()

        G_t = G_imputed.contiguous()

        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            if keep_maf.sum().item() == 0:
                continue
            mask_cpu = keep_maf.detach().cpu().numpy()
            G_t = G_t[keep_maf].contiguous()
            v_idx = v_idx[mask_cpu]
            if H_t is not None:
                H_t = H_t[keep_maf].contiguous()

        pH = 0 if H_t is None else int(H_t.shape[2])
        perm_chunk_local = _auto_perm_chunk(G_t.shape[0], nperm, pH=pH)
        if perm_chunk is not None and perm_chunk > 0:
            perm_chunk_local = min(perm_chunk_local, int(perm_chunk))

        n_samples = ig.phenotype_df.shape[1]
        if perm_ix_t is not None:
            perm_ix_group = roll_for_key(perm_ix_t, f"group:{group_id}", seed)
        else:
            local_seed = subseed(seed, f"group:{group_id}") if seed is not None else None
            perm_ix_group = make_perm_ix(n_samples, nperm, device, local_seed)

        dosage_dict: dict[str, torch.Tensor] = {}
        if seed_vid in var_in_frame and seed_vid in geno_has_variant:
            dosage_dict[seed_vid] = torch.as_tensor(
                dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=seed_vid,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                ),
                dtype=torch.float32,
                device=device,
            )

        ids_list = list(ids)
        y_stack = torch.stack([
            to_device_tensor(
                ig.phenotype_df.loc[pid].to_numpy(np.float32, copy=False), device,
                dtype=torch.float32
            )
            for pid in ids_list
        ], dim=0)

        forward_records: list[dict[str, object]] = []
        base_record = {col: seed_row.get(col) for col in expected_columns if col not in ("rank", "group_size")}
        base_record["group_size"] = len(ids_list)
        base_record["group_id"] = group_id
        forward_records.append(base_record)

        while True:
            extras = [dosage_dict[v] for v in dosage_dict]
            rez_aug = None
            best_ix_var = -1
            best_ix_pheno = -1
            best_beta = best_se = best_t = None
            best_dof = 0
            best_r2_val = -np.inf
            stop_pval = float("inf")
            pval_perm = float("nan")
            pval_beta = float("nan")
            a_hat = float("nan")
            b_hat = float("nan")
            pval_nominal = float("nan")
            num_var = 0
            with torch.no_grad():
                if covariates_base_t is not None or extras:
                    components = []
                    if covariates_base_t is not None:
                        components.append(covariates_base_t)
                    if extras:
                        components.append(torch.stack(extras, dim=1))
                    C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                    rez_aug = Residualizer(C_aug_t, tensorqtl_flavor=tensorqtl_flavor)

                y_resid_list, G_resid, H_resid = residualize_batch(
                    y_stack, G_t, H_t, rez_aug, center=True, group=True
                )
                k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                num_var = int(G_resid.shape[0])

                r2_perm_global = torch.full((int(perm_ix_group.shape[0]),), -float("inf"),
                            device=G_t.device, dtype=torch.float32)
                best_r2_t = torch.tensor(float("-inf"), device=G_t.device)

                for j, (pid_inner, y_resid) in enumerate(zip(ids_list, y_resid_list)):
                    betas, ses, tstats, r2_perm_vec = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=G_resid,
                        H_resid=H_resid,
                        k_eff=k_eff,
                        perm_ix=perm_ix_group,
                        device=device,
                        perm_chunk=perm_chunk_local,
                        return_nominal=True,
                    )
                    p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                    dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                    t_g = tstats[:, 0].double()
                    t2 = t_g.pow(2)
                    r2_nominal_vec = (t2 / (t2 + float(dof))).to(torch.float32)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    if random_tiebreak:
                        ties = torch.nonzero(
                            torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12),
                            as_tuple=True,
                        )[0]
                        if ties.numel() > 1:
                            choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                            ix = int(ties[int(choice_tensor.item())].item())
                            r2_max_t = r2_nominal_vec[ix]
                    if r2_max_t > best_r2_t:
                        best_r2_t = r2_max_t
                        best_r2_val = float(r2_max_t.item())
                        best_ix_var = ix
                        best_ix_pheno = j
                        best_beta = float(betas[ix, 0].item())
                        best_se = float(ses[ix, 0].item())
                        best_t = float(t_g[ix].item())
                        best_dof = int(dof)
                    r2_perm_global = torch.maximum(r2_perm_global, r2_perm_vec)

                if best_ix_var >= 0:
                    r2_perm_max = r2_perm_global
                    pval_perm = (
                        (r2_perm_max >= best_r2_t).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                    ).item()
                    r2_perm_np = r2_perm_max.detach().cpu().numpy()
                    if beta_approx:
                        pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                            r2_perm_np, best_r2_val, dof_init=best_dof
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
                    if np.isfinite(pval_beta):
                        stop_pval = float(pval_beta)
                    else:
                        stop_pval = pval_perm
                    pval_nominal = float(get_t_pval(best_t, best_dof))
                else:
                    stop_pval = float("inf")

            if best_ix_var < 0 or stop_pval > signif_threshold:
                break

            pid_best = ids_list[best_ix_pheno]
            var_id = idx_to_id[v_idx[best_ix_var]]
            var_pos = int(pos_arr[v_idx[best_ix_var]])
            start_pos = ig.phenotype_start[pid_best]
            end_pos = ig.phenotype_end[pid_best]

            forward_records.append({
                "group_id": group_id,
                "group_size": len(ids_list),
                "phenotype_id": pid_best,
                "variant_id": var_id,
                "start_distance": int(var_pos - start_pos),
                "end_distance": int(var_pos - end_pos),
                "num_var": num_var,
                "slope": best_beta,
                "slope_se": best_se,
                "tstat": best_t,
                "r2_nominal": best_r2_val,
                "pval_nominal": pval_nominal,
                "pval_perm": pval_perm,
                "pval_beta": float(pval_beta),
                "beta_shape1": float(a_hat),
                "beta_shape2": float(b_hat),
                "true_dof": int(best_dof),
                "pval_true_dof": p_true,
            })

            if var_id not in dosage_dict and var_id in var_in_frame and var_id in geno_has_variant:
                dosage_dict[var_id] = torch.as_tensor(
                    dosage_vector_for_covariate(
                        genotype_df=ig.genotype_df,
                        variant_id=var_id,
                        sample_order=ig.phenotype_df.columns,
                        missing=missing,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

        if not forward_records:
            continue

        if len(forward_records) > 1:
            kept_records: list[dict[str, object]] = []
            selected = [rec["variant_id"] for rec in forward_records]

            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]

                extras = [dosage_dict[v] for v in kept]
                rez_aug = None
                best_ix_var = -1
                best_ix_pheno = -1
                best_beta = best_se = best_t = None
                best_dof = 0
                best_r2_val = -np.inf
                pval_perm = float("nan")
                pval_beta = float("nan")
                a_hat = float("nan")
                b_hat = float("nan")
                pval_nominal = float("nan")
                num_var = 0
                stop_pval = float("inf")
                with torch.no_grad():
                    if covariates_base_t is not None or extras:
                        components = []
                        if covariates_base_t is not None:
                            components.append(covariates_base_t)
                        if extras:
                            components.append(torch.stack(extras, dim=1))
                        C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                        rez_aug = Residualizer(C_aug_t, tensorqtl_flavor=tensorqtl_flavor)

                    y_resid_list, G_resid, H_resid = residualize_batch(
                        y_stack, G_t, H_t, rez_aug, center=True, group=True
                    )
                    k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                    num_var = int(G_resid.shape[0])

                    r2_perm_list: list[torch.Tensor] = []
                    best_r2_t = torch.tensor(float("-inf"), device=G_t.device)

                    for j, (pid_inner, y_resid) in enumerate(zip(ids_list, y_resid_list)):
                        betas, ses, tstats, r2_perm_vec = compute_perm_r2_max(
                            y_resid=y_resid,
                            G_resid=G_resid,
                            H_resid=H_resid,
                            k_eff=k_eff,
                            perm_ix=perm_ix_group,
                            device=device,
                            perm_chunk=perm_chunk_local,
                            return_nominal=True,
                        )
                        p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                        dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                        t_g = tstats[:, 0].double()
                        t2 = t_g.pow(2)
                        r2_nominal_vec = (t2 / (t2 + float(dof))).to(torch.float32)
                        r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                        ix = int(ix_t.item())
                        if random_tiebreak:
                            ties = torch.nonzero(
                                torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12),
                                as_tuple=True,
                            )[0]
                            if ties.numel() > 1:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                                ix = int(ties[int(choice_tensor.item())].item())
                                r2_max_t = r2_nominal_vec[ix]
                        if r2_max_t > best_r2_t:
                            best_r2_t = r2_max_t
                            best_r2_val = float(r2_max_t.item())
                            best_ix_var = ix
                            best_ix_pheno = j
                            best_beta = float(betas[ix, 0].item())
                            best_se = float(ses[ix, 0].item())
                            best_t = float(t_g[ix].item())
                            best_dof = int(dof)
                        r2_perm_list.append(r2_perm_vec)

                    if best_ix_var >= 0:
                        r2_perm_max = torch.stack(r2_perm_list, dim=0).max(dim=0).values
                        pval_perm = (
                            (r2_perm_max >= best_r2_t).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                        ).item()
                        r2_perm_np = r2_perm_max.detach().cpu().numpy()
                        if beta_approx:
                            pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                                r2_perm_np, best_r2_val, dof_init=dof
                            )
                        else:
                            pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
                        if np.isfinite(pval_beta):
                            stop_pval = float(pval_beta)
                        else:
                            stop_pval = pval_perm
                        pval_nominal = float(get_t_pval(best_t, best_dof))

                if best_ix_var >= 0 and stop_pval <= signif_threshold:
                    pid_best = ids_list[best_ix_pheno]
                    var_id = idx_to_id[v_idx[best_ix_var]]
                    var_pos = int(pos_arr[v_idx[best_ix_var]])
                    start_pos = ig.phenotype_start[pid_best]
                    end_pos = ig.phenotype_end[pid_best]

                    kept_records.append({
                        "group_id": group_id,
                        "group_size": len(ids_list),
                        "phenotype_id": pid_best,
                        "variant_id": var_id,
                        "start_distance": int(var_pos - start_pos),
                        "end_distance": int(var_pos - end_pos),
                        "num_var": num_var,
                        "slope": best_beta,
                        "slope_se": best_se,
                        "tstat": best_t,
                        "r2_nominal": best_r2_val,
                        "pval_nominal": pval_nominal,
                        "pval_perm": pval_perm,
                        "pval_beta": float(pval_beta),
                        "beta_shape1": float(a_hat),
                        "beta_shape2": float(b_hat),
                        "true_dof": int(best_dof),
                        "pval_true_dof": p_true,
                        "rank": int(rk),
                    })

            if not kept_records:
                continue
            records_to_write = kept_records
        else:
            records_to_write = forward_records

        for rk, rec in enumerate(records_to_write, start=1):
            rec["rank"] = rk
            buffers = ensure_capacity(buffers, cursor, 1)
            write_row(buffers, cursor, rec)
            cursor += 1

        processed += 1
        if (
            logger.verbose
            and total_items
            and progress_interval
            and (processed % progress_interval == 0 or processed == total_items)
        ):
            logger.write(
                f"      processed {processed}/{total_items} {item_label} on {chrom_label}"
            )

    return buffers_to_dataframe(expected_columns, buffers, cursor)


def map_independent(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, cis_df: pd.DataFrame,
        phenotype_df: pd.DataFrame, phenotype_pos_df: pd.DataFrame,
        covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        fdr: float = 0.05, fdr_col: str = "qval", nperm: int = 10_000,
        window: int = 1_000_000, missing: float = -9.0, random_tiebreak: bool = False,
        device: str = "auto", beta_approx: bool = True, perm_chunk: int = 2048,
        seed: int | None = None, logger: SimpleLogger | None = None,
        verbose: bool = True, preload_haplotypes: bool = True,
) -> pd.DataFrame:
    """Entry point: build IG; derive seed/threshold from cis_df; dispatch to grouped/ungrouped core.

    Parameters
    ----------
    preload_haplotypes : bool, default True
        When haplotypes are provided, pre-load them into a contiguous torch.Tensor
        on the requested device to avoid per-batch host<->device transfers.
    """
    device = ("cuda" if (device in ("auto", None) and torch.cuda.is_available())
              else (device if device in ("cuda", "cpu") else "cpu"))
    torch_device = torch.device(device)
    logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
    sync = (torch.cuda.synchronize if device == "cuda" else None)

    if not phenotype_df.index.equals(phenotype_pos_df.index):
        raise ValueError("phenotype_df and phenotype_pos_df must share identical indices")

    # Subset FDR-significant rows and compute threshold (max pval_beta)
    if fdr_col not in cis_df.columns:
        raise ValueError(f"cis_df must contain '{fdr_col}'")
    if "pval_beta" not in cis_df.columns:
        raise ValueError("cis_df must contain 'pval_beta'.")

    signif_df = cis_df[cis_df[fdr_col] <= fdr].copy()
    if signif_df.empty:
        raise ValueError(f"No significant phenotypes at FDR ≤ {fdr} in cis_df[{fdr_col}].")
    signif_threshold = float(np.nanmax(signif_df["pval_beta"].values))

    # Header (tensorQTL-style)
    logger.write("cis-QTL mapping: conditionally independent variants")
    logger.write(f"  * device: {device}")
    logger.write(f"  * {phenotype_df.shape[1]} samples")
    logger.write(f'  * {signif_df.shape[0]}/{cis_df.shape[0]} significant phenotypes')
    logger.write(f"  * {variant_df.shape[0]} variants")
    logger.write(f"  * cis-window: \u00B1{window:,}")
    logger.write(f"  * nperm={nperm:,} (beta_approx={'on' if beta_approx else 'off'})")
    if maf_threshold and maf_threshold > 0:
        logger.write(f"  * applying in-sample {maf_threshold:g} MAF filter")
    if covariates_df is not None:
        logger.write(f"  * {covariates_df.shape[1]} covariates")
    if haplotypes is not None:
        K = int(haplotypes.shape[2])
        preload_flag = preload_haplotypes and haplotypes is not None
        logger.write(
            f"  * including local ancestry channels (K={K}, preload={'on' if preload_flag else 'off'})"
        )

    # Build the appropriate input generator (no residualization up front)
    ig = (
        InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
            loci_df=loci_df, group_s=group_s,
            preload_to_torch=(preload_haplotypes and haplotypes is not None),
            torch_device=torch_device,
        )
        if haplotypes is not None else
        InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
    )
    if ig.n_phenotypes == 0:
        raise ValueError("No valid phenotypes after generator preprocessing.")

    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_independent.")

    n_samples = int(ig.phenotype_df.shape[1])
    perm_ix_t = make_perm_ix(n_samples, nperm, device, seed)

    if group_s is None:
        if "phenotype_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'phenotype_id' for ungrouped mapping.")
        signif_seed_df = signif_df.set_index("phenotype_id", drop=False)
        valid_ids = ig.phenotype_pos_df.index.intersection(signif_seed_df.index)
        phenotype_counts = ig.phenotype_pos_df.loc[valid_ids, "chr"].value_counts().to_dict()
        total_items = int(valid_ids.shape[0])
        item_label = "phenotypes"

        def run_core(chrom: str | None, chrom_total: int | None) -> pd.DataFrame:
            return _run_independent_core(
                ig=ig, variant_df=variant_df, covariates_df=covariates_df,
                signif_seed_df=signif_seed_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
                perm_ix_t=perm_ix_t, perm_chunk=perm_chunk,
                logger=logger, total_items=chrom_total, item_label=item_label,
            )
    else:
        if "group_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'group_id' for grouped mapping.")
        seed_by_group_df = (signif_df.sort_values(["group_id", "pval_beta"])
                                      .groupby("group_id", sort=False).head(1))
        group_counts: dict[str, int] = {}
        total_items = 0
        for _, row in seed_by_group_df.iterrows():
            pid = row.get("phenotype_id")
            if pd.isna(pid) or pid not in ig.phenotype_pos_df.index:
                continue
            chrom = ig.phenotype_pos_df.at[pid, "chr"]
            group_counts[chrom] = group_counts.get(chrom, 0) + 1
            total_items += 1
        phenotype_counts = group_counts
        item_label = "phenotype groups"

        def run_core(chrom: str | None, chrom_total: int | None) -> pd.DataFrame:
            return _run_independent_core_group(
                ig=ig, variant_df=variant_df, covariates_df=covariates_df,
                seed_by_group_df=seed_by_group_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
                perm_ix_t=perm_ix_t, perm_chunk=perm_chunk,
                logger=logger, total_items=chrom_total, item_label=item_label,
            )

    if logger.verbose:
        logger.write(f"    Mapping all chromosomes ({total_items} {item_label})")

    overall_start = time.time()
    results: list[pd.DataFrame] = []
    with logger.time_block("Computing associations (independent: forward–backward)", sync=sync):
        for chrom in ig.chrs:
            chrom_total = int(phenotype_counts.get(chrom, 0))
            if logger.verbose:
                logger.write(f"    Mapping chromosome {chrom} ({chrom_total} {item_label})")
            chrom_start = time.time()
            with logger.time_block(f"{chrom}: map_independent", sync=sync):
                chrom_df = run_core(chrom, chrom_total)
            results.append(chrom_df)
            if logger.verbose:
                elapsed = time.time() - chrom_start
                logger.write(f"    Chromosome {chrom} completed in {elapsed:.2f}s")

    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed independent scan in {elapsed:.2f}s")

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    return pd.DataFrame()
