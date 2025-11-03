import time

import torch
import numpy as np
import pandas as pd
from typing import Optional

from ..utils import SimpleLogger, subseed
from ..stats import beta_approx_pval, get_t_pval
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, allele_stats, filter_by_maf
from ..regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .common import residualize_batch, dosage_vector_for_covariate

__all__ = [
    "map_independent",
]


def _allocate_result_buffers(
        columns: list[str], dtype_map: dict[str, type | np.dtype], target_rows: int) -> dict[str, np.ndarray]:
    target = max(int(target_rows), 1)
    buffers: dict[str, np.ndarray] = {}
    for name in columns:
        dtype = dtype_map.get(name, np.float32)
        buffers[name] = np.empty(target, dtype=dtype)
    return buffers


def _ensure_capacity(buffers: dict[str, np.ndarray], cursor: int, additional: int) -> dict[str, np.ndarray]:
    required = cursor + int(additional)
    current = next(iter(buffers.values())).shape[0] if buffers else 0
    if required <= current:
        return buffers
    new_size = max(required, current * 2 if current else 1)
    for key, arr in buffers.items():
        new_arr = np.empty(new_size, dtype=arr.dtype)
        new_arr[:cursor] = arr[:cursor]
        buffers[key] = new_arr
    return buffers


def _write_row(buffers: dict[str, np.ndarray], idx: int, row: dict[str, object]) -> None:
    for key, arr in buffers.items():
        value = row.get(key)
        if arr.dtype == object:
            arr[idx] = None if value is None else value
        elif np.issubdtype(arr.dtype, np.integer):
            arr[idx] = int(value) if value is not None else 0
        else:
            arr[idx] = np.nan if value is None else float(value)


def _buffers_to_dataframe(columns: list[str], buffers: dict[str, np.ndarray], n_rows: int) -> pd.DataFrame:
    n = int(n_rows)
    data = {col: buffers[col][:n] for col in columns if col in buffers}
    return pd.DataFrame(data, columns=columns)

def _run_independent_core(
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        signif_seed_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True, seed: int | None = None,
        chrom: str | None = None
) -> pd.DataFrame:
    """Forward–backward independent mapping for ungrouped phenotypes."""
    expected_columns = [
        "phenotype_id", "variant_id", "pos", "start_distance", "end_distance",
        "num_var", "beta", "se", "tstat", "r2_nominal", "pval_nominal",
        "pval_perm", "pval_beta", "beta_shape1", "beta_shape2", "af",
        "ma_samples", "ma_count", "dof", "rank",
    ]
    dtype_map = {
        "phenotype_id": object,
        "variant_id": object,
        "pos": np.int32,
        "start_distance": np.int32,
        "end_distance": np.int32,
        "num_var": np.int32,
        "beta": np.float32,
        "se": np.float32,
        "tstat": np.float32,
        "r2_nominal": np.float32,
        "pval_nominal": np.float32,
        "pval_perm": np.float32,
        "pval_beta": np.float32,
        "beta_shape1": np.float32,
        "beta_shape2": np.float32,
        "af": np.float32,
        "ma_samples": np.int32,
        "ma_count": np.float32,
        "dof": np.int32,
        "rank": np.int32,
    }
    buffers = _allocate_result_buffers(expected_columns, dtype_map, signif_seed_df.shape[0])
    cursor = 0

    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int32)

    # Basic alignment checks
    covariates_base_t: torch.Tensor | None = None
    if covariates_df is not None:
        if not covariates_df.index.equals(ig.phenotype_df.columns):
            covariates_df = covariates_df.loc[ig.phenotype_df.columns]
        covariates_base_t = torch.as_tensor(
            covariates_df.to_numpy(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
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
        y_t = torch.tensor(p, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        if H_block is None:
            H_t = None
        elif isinstance(H_block, torch.Tensor):
            if H_block.device.type != device:
                H_t = H_block.to(device=device, dtype=torch.float32)
            elif H_block.dtype != torch.float32:
                H_t = H_block.to(dtype=torch.float32)
            else:
                H_t = H_block
        else:
            H_t = torch.tensor(H_block, dtype=torch.float32, device=device)

        # Impute & filter (and optional MAF)
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.sum().item() == 0:
            continue
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1]  # drop one ancestry channel

        # Minor-allele stats (pre-residualization)
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Build per-phenotype generator
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(subseed(seed, pid))

        # Fixed permutations for this phenotype
        n = y_t.shape[0]
        perms = torch.stack(
            [torch.randperm(n, device=device, generator=gen) for _ in range(nperm)],
            dim=1,
        )  # (n, nperm)

        # Forward pass
        forward_rows = [seed_row.to_frame().T]  # initialize with seed from cis_df
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
            # Build augmented covariates = [C | selected dosages]
            rez_aug = None
            extras = [dosage_dict[v] for v in dosage_dict]
            if covariates_base_t is not None or extras:
                components = []
                if covariates_base_t is not None:
                    components.append(covariates_base_t)
                if extras:
                    components.append(torch.stack(extras, dim=1))
                C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                rez_aug = Residualizer(C_aug_t)
            y_resid, G_resid, H_resid = residualize_batch(y_t, G_t, H_t, rez_aug, center=True, group=False)
            y_perm = y_resid[perms]

            k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
            betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
            )

            p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
            dof = max(int(n) - int(k_eff) - int(p_pred), 1)
            t_g = tstats[:, 0]
            r2_nominal_vec = (t_g.double().pow(2) / (t_g.double().pow(2) + dof))
            r2_np = r2_nominal_vec.detach().cpu().numpy()
            r2_max = np.nanmax(r2_np)
            if random_tiebreak:
                ties = np.flatnonzero(np.isclose(r2_np, r2_max, atol=1e-12))
                if gen is None:
                    choice = int(torch.randint(0, len(ties), (1,)).item())
                else:
                    choice = int(torch.randint(0, len(ties), (1,), generator=gen).item())
                ix = int(ties[choice])
            else:
                ix = int(np.nanargmax(r2_np))

            # Stats for the selected variant
            beta = float(betas[ix, 0].detach().cpu().numpy())
            se = float(ses[ix, 0].detach().cpu().numpy())
            tval = float(t_g[ix].detach().cpu().numpy())
            r2_nom = float(r2_np[ix])

            r2_perm_np = r2_perm.detach().cpu().numpy()
            pval_perm = float((np.sum(r2_perm_np >= r2_nom) + 1) / (r2_perm_np.size + 1))
            pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_np, r2_nom) if beta_approx
                                       else (np.nan, np.nan, np.nan))
            stop_pval = float(pval_beta)
            if not np.isfinite(stop_pval):
                stop_pval = pval_perm
            pval_nominal = float(get_t_pval(tval, dof))

            # Stop if not significant under threshold
            if stop_pval > signif_threshold:
                break

            var_id = idx_to_id[v_idx[ix]]
            var_pos = int(pos_arr[v_idx[ix]])
            start_pos = ig.phenotype_start[pid]
            end_pos = ig.phenotype_end[pid]
            start_distance = int(var_pos - start_pos)
            end_distance = int(var_pos - end_pos)
            num_var = int(G_resid.shape[0])
            forward_records.append({
                "phenotype_id": pid,
                "variant_id": var_id,
                "pos": var_pos,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "num_var": num_var,
                "beta": beta,
                "se": se,
                "tstat": tval,
                "r2_nominal": r2_nom,
                "pval_nominal": pval_nominal,
                "pval_perm": pval_perm,
                "pval_beta": float(pval_beta),
                "beta_shape1": float(a_hat),
                "beta_shape2": float(b_hat),
                "af": float(af_t[ix].detach().cpu().numpy()),
                "ma_samples": int(ma_samples_t[ix].detach().cpu().numpy()),
                "ma_count": float(ma_count_t[ix].detach().cpu().numpy()),
                "dof": int(dof),
            })

            # add dosage covariate for next round
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
        for rk, rec in enumerate(forward_records, start=1):
            rec["rank"] = rk

        # Backward pass
        if len(forward_records) > 1:
            kept_records: list[dict[str, object]] = []
            selected = [rec["variant_id"] for rec in forward_records]
            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]
                rez_aug = None
                extras = [dosage_dict[v] for v in kept]
                if covariates_base_t is not None or extras:
                    components = []
                    if covariates_base_t is not None:
                        components.append(covariates_base_t)
                    if extras:
                        components.append(torch.stack(extras, dim=1))
                    C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                    rez_aug = Residualizer(C_aug_t)
                y_resid, G_resid, H_resid = residualize_batch(y_t, G_t, H_t, rez_aug, center=True, group=False)
                y_perm = y_resid[perms]

                k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                    y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                )
                p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                t_g = tstats[:, 0]
                r2_np = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).detach().cpu().numpy()
                r2_max = np.nanmax(r2_np)
                if random_tiebreak:
                    ties = np.flatnonzero(np.isclose(r2_np, r2_max, atol=1e-12))
                    if gen is None:
                        choice = int(torch.randint(0, len(ties), (1,)).item())
                    else:
                        choice = int(torch.randint(0, len(ties), (1,), generator=gen).item())
                    ix = int(ties[choice])
                else:
                    ix = int(np.nanargmax(r2_np))

                beta = float(betas[ix, 0].detach().cpu().numpy())
                se = float(ses[ix, 0].detach().cpu().numpy())
                tval = float(t_g[ix].detach().cpu().numpy())
                r2_nom = float(r2_np[ix])

                r2_perm_np = r2_perm.detach().cpu().numpy()
                pval_perm = float((np.sum(r2_perm_np >= r2_nom) + 1) / (r2_perm_np.size + 1))
                pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_np, r2_nom) if beta_approx
                                           else (np.nan, np.nan, np.nan))
                stop_pval = float(pval_beta)
                if not np.isfinite(stop_pval):
                    stop_pval = pval_perm

                if stop_pval <= signif_threshold:
                    var_id = idx_to_id[v_idx[ix]]
                    var_pos = int(pos_arr[v_idx[ix]])
                    start_pos = ig.phenotype_start[pid]
                    end_pos = ig.phenotype_end[pid]
                    kept_records.append({
                        "phenotype_id": pid,
                        "variant_id": var_id,
                        "pos": var_pos,
                        "start_distance": int(var_pos - start_pos),
                        "end_distance": int(var_pos - end_pos),
                        "num_var": int(G_resid.shape[0]),
                        "beta": beta,
                        "se": se,
                        "tstat": tval,
                        "r2_nominal": r2_nom,
                        "pval_nominal": float(get_t_pval(tval, dof)),
                        "pval_perm": pval_perm,
                        "pval_beta": float(pval_beta),
                        "beta_shape1": float(a_hat),
                        "beta_shape2": float(b_hat),
                        "af": float(af_t[ix].detach().cpu().numpy()),
                        "ma_samples": int(ma_samples_t[ix].detach().cpu().numpy()),
                        "ma_count": float(ma_count_t[ix].detach().cpu().numpy()),
                        "dof": int(dof),
                        "rank": int(rk),
                    })

            if not kept_records:
                continue
            records_to_write = kept_records
        else:
            records_to_write = forward_records

        for rec in records_to_write:
            buffers = _ensure_capacity(buffers, cursor, 1)
            _write_row(buffers, cursor, rec)
            cursor += 1

    return _buffers_to_dataframe(expected_columns, buffers, cursor)



def _run_independent_core_group( ## TODO: Needs updating
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        seed_by_group_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True, seed: int | None = None,
        chrom: str | None = None
) -> pd.DataFrame:
    """Forward–backward independent mapping for grouped phenotypes."""
    expected_columns = [
        "group_id", "group_size", "phenotype_id", "variant_id", "pos",
        "start_distance", "end_distance", "num_var", "beta", "se", "tstat",
        "r2_nominal", "pval_nominal", "pval_perm", "pval_beta", "beta_shape1",
        "beta_shape2", "dof", "rank",
    ]
    dtype_map = {
        "group_id": object,
        "group_size": np.int32,
        "phenotype_id": object,
        "variant_id": object,
        "pos": np.int32,
        "start_distance": np.int32,
        "end_distance": np.int32,
        "num_var": np.int32,
        "beta": np.float32,
        "se": np.float32,
        "tstat": np.float32,
        "r2_nominal": np.float32,
        "pval_nominal": np.float32,
        "pval_perm": np.float32,
        "pval_beta": np.float32,
        "beta_shape1": np.float32,
        "beta_shape2": np.float32,
        "dof": np.int32,
        "rank": np.int32,
    }
    buffers = _allocate_result_buffers(expected_columns, dtype_map, seed_by_group_df.shape[0])
    cursor = 0

    var_in_frame = set(variant_df.index)
    geno_has_variant = set(ig.genotype_df.index)
    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int64, copy=False)

    covariates_base_t: torch.Tensor | None = None
    if covariates_df is not None:
        if not covariates_df.index.equals(ig.phenotype_df.columns):
            covariates_df = covariates_df.loc[ig.phenotype_df.columns]
        covariates_base_t = torch.as_tensor(
            covariates_df.to_numpy(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
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

        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        if H_block is None:
            H_t = None
        elif isinstance(H_block, torch.Tensor):
            if H_block.device.type != device:
                H_t = H_block.to(device=device, dtype=torch.float32)
            elif H_block.dtype != torch.float32:
                H_t = H_block.to(dtype=torch.float32)
            else:
                H_t = H_block
        else:
            H_t = torch.tensor(H_block, dtype=torch.float32, device=device)

        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.sum().item() == 0:
            continue
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]
            if H_t.shape[2] > 1:
                H_t = H_t[:, :, :-1]

        n = ig.phenotype_df.shape[1]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(subseed(seed, f"group:{group_id}"))
        perms = torch.stack(
            [torch.randperm(n, device=device, generator=gen) for _ in range(nperm)],
            dim=1,
        )

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
        forward_records: list[dict[str, object]] = []
        base_record = {col: seed_row.get(col) for col in expected_columns if col not in ("rank", "group_size")}
        base_record["group_size"] = len(ids)
        base_record["group_id"] = group_id
        forward_records.append(base_record)

        while True:
            # augmented covariates for this forward step
            rez_aug = None
            extras = [dosage_dict[v] for v in dosage_dict]
            if covariates_base_t is not None or extras:
                components = []
                if covariates_base_t is not None:
                    components.append(covariates_base_t)
                if extras:
                    components.append(torch.stack(extras, dim=1))
                C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                rez_aug = Residualizer(C_aug_t)

            best = dict(r2=-np.inf, ix_var=-1, ix_pheno=-1, beta=None, se=None, t=None, dof=None)
            r2_perm_list = []

            mats = [G_t]
            H_shape = None
            if H_t is not None:
                m, n_s, pH = H_t.shape
                H_shape = (m, n_s, pH)
                mats.append(H_t.reshape(m * pH, n_s))

            for j, pid in enumerate(ids):
                y = ig.phenotype_df.loc[pid].to_numpy(np.float32)
                y_t = torch.tensor(y, dtype=torch.float32, device=device)

                if rez_aug is None:
                    mats_resid = [G_t] + ([H_t.reshape(H_shape[0] * H_shape[2], H_shape[1])] if H_t is not None else []) + [y_t.unsqueeze(0)]
                else:
                    mats_resid = rez_aug.transform(*mats, y_t.unsqueeze(0), center=True)
                G_resid = mats_resid[0]
                idx = 1
                H_resid = None
                if H_t is not None:
                    H_resid = mats_resid[idx].reshape(H_shape)
                    idx += 1
                y_resid = mats_resid[idx].squeeze(0)

                y_perm = y_resid[perms]
                k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                    y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                )
                p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                t_g = tstats[:, 0]
                r2_nominal_vec = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).detach().cpu().numpy()
                ix = int(np.nanargmax(r2_nominal_vec))
                if r2_nominal_vec[ix] > best["r2"]:
                    best.update(
                        r2=float(r2_nominal_vec[ix]),
                        ix_var=ix,
                        ix_pheno=j,
                        beta=float(betas[ix, 0].detach().cpu().numpy()),
                        se=float(ses[ix, 0].detach().cpu().numpy()),
                        t=float(t_g[ix].detach().cpu().numpy()),
                        dof=int(dof),
                    )
                r2_perm_list.append(r2_perm)

            r2_perm_max = torch.stack(r2_perm_list, dim=0).max(dim=0).values.detach().cpu().numpy()
            pval_perm = float((np.sum(r2_perm_max >= best["r2"]) + 1) / (r2_perm_max.size + 1))
            pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_max, best["r2"]) if beta_approx
                                       else (np.nan, np.nan, np.nan))
            stop_pval = float(pval_beta)
            if not np.isfinite(stop_pval):
                stop_pval = pval_perm
            if stop_pval > signif_threshold:
                break

            pid_best = ids[best["ix_pheno"]]
            var_id = idx_to_id[v_idx[best["ix_var"]]]
            var_pos = int(pos_arr[v_idx[best["ix_var"]]])
            start_pos = ig.phenotype_start[pid_best]
            end_pos = ig.phenotype_end[pid_best]

            forward_records.append({
                "group_id": group_id,
                "group_size": len(ids),
                "phenotype_id": pid_best,
                "variant_id": var_id,
                "pos": var_pos,
                "start_distance": int(var_pos - start_pos),
                "end_distance": int(var_pos - end_pos),
                "num_var": int(G_t.shape[0]),
                "beta": best["beta"],
                "se": best["se"],
                "tstat": best["t"],
                "r2_nominal": best["r2"],
                "pval_nominal": float(get_t_pval(best["t"], best["dof"])),
                "pval_perm": pval_perm,
                "pval_beta": float(pval_beta),
                "beta_shape1": float(a_hat),
                "beta_shape2": float(b_hat),
                "dof": int(best["dof"]),
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
        for rk, rec in enumerate(forward_records, start=1):
            rec["rank"] = rk

        if len(forward_records) > 1:
            kept_records: list[dict[str, object]] = []
            selected = [rec["variant_id"] for rec in forward_records]

            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]

                rez_aug = None
                extras = [dosage_dict[v] for v in kept]
                if covariates_base_t is not None or extras:
                    components = []
                    if covariates_base_t is not None:
                        components.append(covariates_base_t)
                    if extras:
                        components.append(torch.stack(extras, dim=1))
                    C_aug_t = torch.cat(components, dim=1) if len(components) > 1 else components[0]
                    rez_aug = Residualizer(C_aug_t)

                best = dict(r2=-np.inf, ix_var=-1, ix_pheno=-1, beta=None, se=None, t=None, dof=None)
                r2_perm_list = []
                mats = [G_t]
                H_shape = None
                if H_t is not None:
                    m, n_s, pH = H_t.shape
                    H_shape = (m, n_s, pH)
                    mats.append(H_t.reshape(m * pH, n_s))

                for j, pid in enumerate(ids):
                    y = ig.phenotype_df.loc[pid].to_numpy(np.float32)
                    y_t = torch.tensor(y, dtype=torch.float32, device=device)
                    if rez_aug is None:
                        mats_resid = [G_t] + ([H_t.reshape(H_shape[0] * H_shape[2], H_shape[1])] if H_t is not None else []) + [y_t.unsqueeze(0)]
                    else:
                        mats_resid = rez_aug.transform(*mats, y_t.unsqueeze(0), center=True)
                    G_resid = mats_resid[0]
                    idx = 1
                    H_resid = None
                    if H_t is not None:
                        H_resid = mats_resid[idx].reshape(H_shape)
                        idx += 1
                    y_resid = mats_resid[idx].squeeze(0)

                    y_perm = y_resid[perms]
                    k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                    betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                        y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                    )
                    p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                    dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                    t_g = tstats[:, 0]
                    r2_nominal_vec = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).detach().cpu().numpy()
                    ix = int(np.nanargmax(r2_nominal_vec))
                    if r2_nominal_vec[ix] > best["r2"]:
                        best.update(
                            r2=float(r2_nominal_vec[ix]),
                            ix_var=ix,
                            ix_pheno=j,
                            beta=float(betas[ix, 0].detach().cpu().numpy()),
                            se=float(ses[ix, 0].detach().cpu().numpy()),
                            t=float(t_g[ix].detach().cpu().numpy()),
                            dof=int(dof),
                        )
                    r2_perm_list.append(r2_perm)

                r2_perm_max = torch.stack(r2_perm_list, dim=0).max(dim=0).values.detach().cpu().numpy()
                pval_perm = float((np.sum(r2_perm_max >= best["r2"]) + 1) / (r2_perm_max.size + 1))
                pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_max, best["r2"]) if beta_approx
                                           else (np.nan, np.nan, np.nan))
                stop_pval = float(pval_beta)
                if not np.isfinite(stop_pval):
                    stop_pval = pval_perm

                if stop_pval <= signif_threshold:
                    pid_best = ids[best["ix_pheno"]]
                    var_id = idx_to_id[v_idx[best["ix_var"]]]
                    var_pos = int(pos_arr[v_idx[best["ix_var"]]])
                    start_pos = ig.phenotype_start[pid_best]
                    end_pos = ig.phenotype_end[pid_best]

                    kept_records.append({
                        "group_id": group_id,
                        "group_size": len(ids),
                        "phenotype_id": pid_best,
                        "variant_id": var_id,
                        "pos": var_pos,
                        "start_distance": int(var_pos - start_pos),
                        "end_distance": int(var_pos - end_pos),
                        "num_var": int(G_t.shape[0]),
                        "beta": best["beta"],
                        "se": best["se"],
                        "tstat": best["t"],
                        "r2_nominal": best["r2"],
                        "pval_nominal": float(get_t_pval(best["t"], best["dof"])),
                        "pval_perm": pval_perm,
                        "pval_beta": float(pval_beta),
                        "beta_shape1": float(a_hat),
                        "beta_shape2": float(b_hat),
                        "dof": int(best["dof"]),
                        "rank": int(rk),
                    })

            if not kept_records:
                continue
            records_to_write = kept_records
        else:
            records_to_write = forward_records

        for rec in records_to_write:
            buffers = _ensure_capacity(buffers, cursor, 1)
            _write_row(buffers, cursor, rec)
            cursor += 1

    return _buffers_to_dataframe(expected_columns, buffers, cursor)
def map_independent(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, cis_df: pd.DataFrame,
        phenotype_df: pd.DataFrame, phenotype_pos_df: pd.DataFrame,
        covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        fdr: float = 0.05, fdr_col: str = "qval", nperm: int = 10_000,
        window: int = 1_000_000, missing: float = -9.0, random_tiebreak: bool = False,
        device: str = "auto", beta_approx: bool = True, seed: int | None = None,
        logger: SimpleLogger | None = None, verbose: bool = True,
        preload_haplotypes: bool = True,
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

    if group_s is None:
        if "phenotype_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'phenotype_id' for ungrouped mapping.")
        signif_seed_df = signif_df.set_index("phenotype_id", drop=False)
        valid_ids = ig.phenotype_pos_df.index.intersection(signif_seed_df.index)
        phenotype_counts = ig.phenotype_pos_df.loc[valid_ids, "chr"].value_counts().to_dict()
        total_items = int(valid_ids.shape[0])
        item_label = "phenotypes"

        def run_core(chrom: str | None) -> pd.DataFrame:
            return _run_independent_core(
                ig=ig, variant_df=variant_df, covariates_df=covariates_df,
                signif_seed_df=signif_seed_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
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

        def run_core(chrom: str | None) -> pd.DataFrame:
            return _run_independent_core_group(
                ig=ig, variant_df=variant_df, covariates_df=covariates_df,
                seed_by_group_df=seed_by_group_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
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
                chrom_df = run_core(chrom)
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
