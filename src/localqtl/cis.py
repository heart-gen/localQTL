import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .utils import SimpleLogger
from .iosinks import ParquetSink
from .haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from .regression_kernels import (
    Residualizer,
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .stats import beta_approx_pval, get_t_pval, nominal_pvals_tensorqtl
from .preproc import impute_mean_and_filter, allele_stats, filter_by_maf

def _dosage_vector_for_covariate(genotype_df: pd.DataFrame, variant_id: str,
                                 sample_order: pd.Index, missing: float | int | None) -> np.ndarray:
    """Fetch a dosage row aligned to samples; impute 'missing' to mean of observed."""
    if not genotype_df.columns.equals(sample_order):
        row = genotype_df.loc[variant_id, sample_order].to_numpy(dtype=np.float32, copy=True)
    else:
        row = genotype_df.loc[variant_id].to_numpy(dtype=np.float32, copy=True)
    if missing is not None:
        mm = (row == missing)
        if mm.any():
            if (~mm).any():
                row[mm] = row[~mm].mean(dtype=np.float32)
            else:
                row[mm] = 0.0
    return row


def _residualize_matrix_with_covariates(
        Y: torch.Tensor, C: Optional[pd.DataFrame], device: str
) -> Tuple[torch.Tensor, Optional[Residualizer]]:
    """
    Residualize (features x samples) matrix Y against covariates C across samples.
    Returns residualized Y and a Residualizer (or None if no covariates).
    """
    if C is None:
        return Y, None
    C_t = torch.tensor(C.values, dtype=torch.float32, device=device)
    rez = Residualizer(C_t)
    (Y_resid,) = rez.transform(Y, center=True)
    return Y_resid, rez


def _residualize_batch(
        y, G: torch.Tensor, H: Optional[torch.Tensor],
        rez: Optional[Residualizer], center: bool = True, group: bool = False,
) -> Tuple[object, torch.Tensor, Optional[torch.Tensor]]:
    """
    Residualize y, G, and optional H with the same Residualizer.
    - If group=False: y is a single vector (n,), returns (y_resid 1D, G_resid, H_resid).
    - If group=True:  y is a stack (k x n) or list of length k, returns (list_of_k 1D tensors, G_resid, H_resid).
    """
    dev = (rez.Q_t.device if (rez is not None and hasattr(rez, "Q_t")) else G.device)
    G = G.to(dev)
    if H is not None:
        H = H.to(dev)

    if rez is None:
        # Pass-through; normalize return type for group mode
        if group:
            if isinstance(y, (list, tuple)):
                y_list = [torch.as_tensor(yi) if not torch.is_tensor(yi) else yi
                          for yi in y]
            else:
                Y = y if torch.is_tensor(y) else torch.as_tensor(y)
                y_list = [Y] if Y.ndim == 1 else [Y[i, :] for i in range(Y.shape[0])]
            return y_list, G, H
        return (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)), G, H

    # Build Y on the same device as G/rez
    if group:
        if isinstance(y, (list, tuple)):
            Y = torch.stack([yi if torch.is_tensor(yi) else torch.as_tensor(yi, dtype=torch.float32, device=dev)
                             for yi in y], dim=0).to(dev)
        else:
            Y = (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32, device=dev))
            if Y.ndim == 1:
                Y = Y.unsqueeze(0)
            Y = Y.to(dev)
    else:
        Y = (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32, device=dev))
        if Y.ndim == 1:
            Y = Y.unsqueeze(0)
        Y = Y.to(dev)
    
    # Prepare matrices for one-pass residualization
    mats: List[torch.Tensor] = [G]
    H_shape = None
    if H is not None:
        m, n, pH = H.shape
        H_shape = (m, n, pH)
        mats.append(H.reshape(m * pH, n))

    mats_resid = rez.transform(*mats, Y, center=center)

    G_resid = mats_resid[0]
    idx, H_resid = 1, None
    if H is not None:
        H_resid = mats_resid[idx].reshape(H_shape)
        idx += 1
    Y_resid = mats_resid[idx]

    if group:
        return [Y_resid[i, :] for i in range(Y_resid.shape[0])], G_resid, H_resid
    else:
        return Y_resid.squeeze(0), G_resid, H_resid


def _run_nominal_core(ig, variant_df, rez, nperm, device, maf_threshold: float = 0.0):
    """
    Shared inner loop for nominal (and optional permutation) mapping.
    Handles both InputGeneratorCis (no haps) and InputGeneratorCisWithHaps (haps).
    """
    out_rows = []
    group_mode = getattr(ig, "group_s", None) is not None
    for batch in ig.generate_data():
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
        y_resid, G_resid, H_resid = _residualize_batch(
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
                y_perm = torch.stack([y_t[idx] for idx in perms], dim=1) # (n x nperm)
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
            out_rows.append(df)

    return pd.concat(out_rows, axis=0).reset_index(drop=True)


def _run_permutation_core(ig, variant_df, rez, nperm: int, device: str,
                          beta_approx: bool = True, maf_threshold: float = 0.0) -> pd.DataFrame:
    """
    One top association per phenotype with empirical permutation p-value (no grouping).
    Compatible with InputGeneratorCis and InputGeneratorCisWithHaps (ungrouped only).
    """
    out_rows = []
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    for batch in ig.generate_data():
        # Accept shapes: (p, G, v_idx, pid) or (p, G, v_idx, H, pid)
        if len(batch) == 4:
            p, G_block, v_idx, pid = batch
            H_block = None
        elif len(batch) == 5 and not isinstance(batch[3], (list, tuple)):
            p, G_block, v_idx, H_block, pid = batch
        else:
            # Skip groups in this core (keep parity with tensorQTL's per-phenotype map_cis)
            raise ValueError("Group mode not supported in _run_permutation_core.")

        # Tensors
        y_t = torch.tensor(p, dtype=torch.float32, device=device)
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Impute and drop monomorphic
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
            continue

        # Optional MAF filter
        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            keep_mask = keep_mask & keep_maf
            if keep_mask.sum().item() == 0:
                continue

        # Mask G, H, and indices consistently
        G_t = G_t[keep_mask]
        v_idx = v_idx[keep_mask.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mask]

        # Minor-allele stats before residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Residualize y/G/H against covariates
        y_t, G_t, H_t = _residualize_batch(y_t, G_t, H_t, rez, center=True)

        # Compute effective covariate rank for DoF
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Build permuted phenotypes (n x nperm)
        perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
        y_perm = torch.stack([y_t[idx] for idx in perms], dim=1)
        
        # Run batched regression with permutations
        betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
            y=y_t, G=G_t, H=H_t, y_perm=y_perm, k_eff=k_eff, device=device
        )

        # Choose top variant by partial R^2 for genotype predictor
        # For multiple regression, partial R^2 for a single predictor:
        # R^2_partial = t^2 / (t^2 + dof), dof = n - p
        n = y_t.shape[0]
        p_pred = 1 + (H_t.shape[2] if H_t is not None else 0)   # genotype + (k-1) ancestry columns
        dof = max(n - p_pred, 1)
        t_g = tstats[:, 0]                                      # (m,)
        r2_nominal_vec = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).cpu().numpy()
        ix = int(np.nanargmax(r2_nominal_vec))

        # Extract top variant stats
        beta = float(betas[ix, 0].detach().cpu().numpy())
        se = float(ses[ix, 0].detach().cpu().numpy())
        tval = float(tstats[ix, 0].detach().cpu().numpy())
        r2_nominal = float(r2_nominal_vec[ix])

        # Nominal p (two-sided t)
        pval_nominal = float(get_t_pval(tval, dof))

        # Empirical permutation p (max across variants each perm)
        r2_perm_np = r2_perm.detach().cpu().numpy()  # (nperm,)
        pval_perm = float((np.sum(r2_perm_np >= r2_nominal) + 1) / (r2_perm_np.size + 1))

        # Optional Beta approximation
        if beta_approx:
            pval_beta, a_hat, b_hat = beta_approx_pval(r2_perm_np, r2_nominal)
        else:
            pval_beta, a_hat, b_hat = np.nan, np.nan, np.nan

        # Metadata
        var_id = variant_df.index.values[v_idx[ix]]
        var_pos = int(variant_df.iloc[v_idx[ix]]["pos"])
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = int(var_pos - start_pos)
        end_distance = int(var_pos - end_pos)
        num_var = int(G_t.shape[0])

        out_rows.append(pd.Series({
            "phenotype_id": pid,
            "variant_id": var_id,
            "pos": var_pos,
            "start_distance": start_distance,
            "end_distance": end_distance,
            "num_var": num_var,
            "beta": beta,
            "se": se,
            "tstat": tval,
            "r2_nominal": r2_nominal,
            "pval_nominal": pval_nominal,
            "pval_perm": pval_perm,
            "pval_beta": pval_beta,
            "beta_shape1": a_hat,
            "beta_shape2": b_hat,
            "af": float(af_t[ix].detach().cpu().numpy()),
            "ma_samples": int(ma_samples_t[ix].detach().cpu().numpy()),
            "ma_count": float(ma_count_t[ix].detach().cpu().numpy()),
            "dof": dof,
        }))

    return pd.DataFrame(out_rows).reset_index(drop=True)


def _run_permutation_core_group(ig, variant_df, rez, nperm: int, device: str,
                                beta_approx: bool = True, maf_threshold: float = 0.0) -> pd.DataFrame:
    """
    Group-aware permutation mapping: returns one top association per *group* (best phenotype within group),
    with empirical p-values computed by taking the max R² across variants and phenotypes for each permutation.
    Mirrors tensorQTL’s grouped behavior.
    """
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    out_rows = []
    for batch in ig.generate_data():
        # Accept shapes: (P, G, v_idx, ids, group_id) or (P, G, v_idx, H, ids, group_id)
        if len(batch) == 5:
            P, G_block, v_idx, ids, group_id = batch
            H_block = None
        elif len(batch) == 6:
            P, G_block, v_idx, H_block, ids, group_id = batch
        else:
            raise ValueError("Unexpected grouped batch shape.")

        # Tensors for window
        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = torch.tensor(H_block, dtype=torch.float32, device=device) if H_block is not None else None

        # Impute + drop monomorphic
        G_t, keep_mask, _ = impute_mean_and_filter(G_t)
        if keep_mask.numel() == 0 or keep_mask.sum().item() == 0:
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
                H_t = H_t[:, :, :-1]  # drop one ancestry channel to avoid rank deficiency

        # Minor-allele stats prior to residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Prepare phenotype list
        if isinstance(P, (list, tuple)):
            P_list = list(P)
        else:
            P = np.asarray(P)
            P_list = [P[i, :] for i in range(P.shape[0])] if P.ndim == 2 else [P]

        # Residualize once; reuse G/H residuals for each phenotype
        # Build a stacked Y for residualization
        Y_stack = torch.stack([torch.tensor(pi, dtype=torch.float32, device=device) for pi in P_list], dim=0)  # (k x n)
        # Use shared routine to residualize matrices with the same Residualizer
        mats: List[torch.Tensor] = [G_t]
        H_shape = None
        if H_t is not None:
            m, n, pH = H_t.shape
            H_shape = (m, n, pH)
            mats.append(H_t.reshape(m * pH, n))
        mats_resid = rez.transform(*mats, Y_stack, center=True) if rez is not None else [G_t] + ([H_t.reshape(m*pH, n)] if H_t is not None else []) + [Y_stack]
        G_resid = mats_resid[0]
        idx = 1
        H_resid = None
        if H_t is not None:
            H_resid = mats_resid[idx].reshape(H_shape)
            idx += 1
        Y_resid = mats_resid[idx]  # (k x n)

        # Design meta
        n = Y_resid.shape[1]
        p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
        dof = max(n - p_pred, 1)
        var_ids = variant_df.index.values[v_idx]
        var_pos = variant_df.iloc[v_idx]["pos"].values
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Evaluate each phenotype: t-stats -> partial R²; keep the global best (variant, phenotype)
        best = dict(r2=-np.inf, ix_var=-1, ix_pheno=-1, beta=None, se=None, t=None)
        r2_perm_list = []
        for j in range(Y_resid.shape[0]):
            y_t = Y_resid[j, :]
            # Permutations for phenotype j
            perms = [torch.randperm(n, device=device) for _ in range(nperm)]
            y_perm = torch.stack([y_t[idx] for idx in perms], dim=1)  # (n x nperm)

            betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                y=y_t, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
            )
            # nominal partial R² for genotype predictor
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
                    t=float(tstats[ix, 0].detach().cpu().numpy()),
                )
            r2_perm_list.append(r2_perm)  # (nperm,)

        # Combine permutations across phenotypes by elementwise max
        r2_perm_max = torch.stack(r2_perm_list, dim=0).max(dim=0).values.detach().cpu().numpy()  # (nperm,)

        # Build output (metadata for the winning phenotype/variant)
        pid = ids[best["ix_pheno"]]
        var_id = var_ids[best["ix_var"]]
        pos = int(var_pos[best["ix_var"]])
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = int(pos - start_pos)
        end_distance = int(pos - end_pos)
        num_var = int(G_t.shape[0])

        # p-values
        pval_nominal = float(get_t_pval(best["t"], dof))
        pval_perm = float((np.sum(r2_perm_max >= best["r2"]) + 1) / (r2_perm_max.size + 1))
        if beta_approx:
            pval_beta, a_hat, b_hat = beta_approx_pval(r2_perm_max, best["r2"])
        else:
            pval_beta, a_hat, b_hat = np.nan, np.nan, np.nan

        out_rows.append(pd.Series({
            "group_id": group_id,
            "group_size": len(ids),
            "phenotype_id": pid,
            "variant_id": var_id,
            "pos": pos,
            "start_distance": start_distance,
            "end_distance": end_distance,
            "num_var": num_var,
            "beta": best["beta"],
            "se": best["se"],
            "tstat": best["t"],
            "r2_nominal": best["r2"],
            "pval_nominal": pval_nominal,
            "pval_perm": pval_perm,
            "pval_beta": pval_beta,
            "beta_shape1": a_hat,
            "beta_shape2": b_hat,
            "af": float(af_t[best["ix_var"]].detach().cpu().numpy()),
            "ma_samples": int(ma_samples_t[best["ix_var"]].detach().cpu().numpy()),
            "ma_count": float(ma_count_t[best["ix_var"]].detach().cpu().numpy()),
            "dof": dof,
        }))

    return pd.DataFrame(out_rows).reset_index(drop=True)


def _run_independent_core(
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        signif_seed_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True,
) -> pd.DataFrame:
    """Forward–backward independent mapping for ungrouped phenotypes."""
    out_rows = []

    # Basic alignment checks
    if covariates_df is not None and not covariates_df.index.equals(ig.phenotype_df.columns):
        covariates_df = covariates_df.loc[ig.phenotype_df.columns]

    # Precompute variant index lookup for dosages
    var_in_frame = set(variant_df.index)

    for batch in ig.generate_data():
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
        H_t = (None if H_block is None else torch.tensor(H_block, dtype=torch.float32, device=device))

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

        # Fixed permutations for this phenotype
        n = y_t.shape[0]
        perms = torch.stack([torch.randperm(n, device=device) for _ in range(nperm)], dim=0)  # (nperm, n)

        # Forward pass
        forward_rows = [seed_row.to_frame().T]  # initialize with seed from cis_df
        dosage_dict: dict[str, np.ndarray] = {}
        seed_vid = str(seed_row["variant_id"])
        if seed_vid in var_in_frame:
            dosage_dict[seed_vid] = _dosage_vector_for_covariate(
                genotype_df=ig.genotype_df,
                variant_id=seed_vid,
                sample_order=ig.phenotype_df.columns,
                missing=missing,
            )

        while True:
            # Build augmented covariates = [C | selected dosages]
            if covariates_df is not None:
                C_base = covariates_df.values.astype(np.float32, copy=False)
                C_sel = (np.column_stack([dosage_dict[v] for v in dosage_dict])
                         if dosage_dict else np.zeros((n, 0), np.float32))
                C_aug = np.hstack([C_base, C_sel]).astype(np.float32, copy=False)
            else:
                C_aug = (np.column_stack([dosage_dict[v] for v in dosage_dict]) if dosage_dict
                         else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)

            rez_aug = Residualizer(torch.tensor(C_aug, dtype=torch.float32, device=device)) if C_aug.shape[1] > 0 else None
            y_resid, G_resid, H_resid = _residualize_batch(y_t, G_t, H_t, rez_aug, center=True, group=False)
            y_perm = torch.stack([y_resid[idx] for idx in perms], dim=1)  # (n x nperm)

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
            ix = (int(np.random.choice(np.flatnonzero(np.isclose(r2_np, r2_max, atol=1e-12)))))
            if not random_tiebreak:
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
            pval_nominal = float(get_t_pval(tval, dof))

            # Stop if not significant under threshold
            if pval_beta > signif_threshold:
                break

            var_id = variant_df.index.values[v_idx[ix]]
            var_pos = int(variant_df.iloc[v_idx[ix]]["pos"])
            start_pos = ig.phenotype_start[pid]
            end_pos = ig.phenotype_end[pid]
            start_distance = int(var_pos - start_pos)
            end_distance = int(var_pos - end_pos)
            num_var = int(G_resid.shape[0])

            row = pd.Series({
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
            forward_rows.append(row.to_frame().T)

            # add dosage covariate for next round
            if var_id in var_in_frame and var_id not in dosage_dict:
                dosage_dict[var_id] = _dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=var_id,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                )

        forward_df = pd.concat(forward_rows, axis=0, ignore_index=True)
        forward_df["rank"] = np.arange(1, forward_df.shape[0] + 1, dtype=int)

        # Backward pass
        if forward_df.shape[0] > 1:
            kept_rows = []
            selected = forward_df["variant_id"].tolist()
            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]
                if covariates_df is not None:
                    C_base = covariates_df.values.astype(np.float32, copy=False)
                    C_sel = (np.column_stack([dosage_dict[v] for v in kept]) if kept
                             else np.zeros((n, 0), np.float32))
                    C_aug = np.hstack([C_base, C_sel]).astype(np.float32, copy=False)
                else:
                    C_aug = (np.column_stack([dosage_dict[v] for v in kept]) if kept
                             else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)

                rez_aug = Residualizer(torch.tensor(C_aug, dtype=torch.float32, device=device)) if C_aug.shape[1] > 0 else None
                y_resid, G_resid, H_resid = _residualize_batch(y_t, G_t, H_t, rez_aug, center=True, group=False)
                y_perm = torch.stack([y_resid[idx] for idx in perms], dim=1)

                k_eff = rez_aug.Q_t.shape[1] if rez_aug is not None else 0
                betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
                    y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                )
                p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                dof = max(int(n) - int(k_eff) - int(p_pred), 1)
                t_g = tstats[:, 0]
                r2_np = (t_g.double().pow(2) / (t_g.double().pow(2) + dof)).detach().cpu().numpy()
                r2_max = np.nanmax(r2_np)
                ix = (int(np.random.choice(np.flatnonzero(np.isclose(r2_np, r2_max, atol=1e-12)))))
                if not random_tiebreak:
                    ix = int(np.nanargmax(r2_np))

                beta = float(betas[ix, 0].detach().cpu().numpy())
                se = float(ses[ix, 0].detach().cpu().numpy())
                tval = float(t_g[ix].detach().cpu().numpy())
                r2_nom = float(r2_np[ix])

                r2_perm_np = r2_perm.detach().cpu().numpy()
                pval_perm = float((np.sum(r2_perm_np >= r2_nom) + 1) / (r2_perm_np.size + 1))
                pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_np, r2_nom) if beta_approx
                                           else (np.nan, np.nan, np.nan))

                if pval_beta <= signif_threshold:
                    var_id = variant_df.index.values[v_idx[ix]]
                    var_pos = int(variant_df.iloc[v_idx[ix]]["pos"])
                    start_pos = ig.phenotype_start[pid]
                    end_pos = ig.phenotype_end[pid]
                    row = pd.Series({
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
                    kept_rows.append(row.to_frame().T)

            if kept_rows:
                out_rows.append(pd.concat(kept_rows, axis=0, ignore_index=True))
        else:
            out_rows.append(forward_df)

    if not out_rows:
        return pd.DataFrame(columns=[
            "phenotype_id","variant_id","pos","start_distance","end_distance","num_var",
            "beta","se","tstat","r2_nominal","pval_nominal","pval_perm","pval_beta",
            "beta_shape1","beta_shape2","af","ma_samples","ma_count","dof","rank"
        ])
    return pd.concat(out_rows, axis=0, ignore_index=True)


def _run_independent_core_group(
        ig, variant_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame],
        seed_by_group_df: pd.DataFrame, signif_threshold: float, nperm: int,
        device: str, maf_threshold: float = 0.0, random_tiebreak: bool = False,
        missing: float = -9.0, beta_approx: bool = True,
) -> pd.DataFrame:
    """Forward–backward independent mapping for grouped phenotypes."""
    out_rows = []

    if covariates_df is not None and not covariates_df.index.equals(ig.phenotype_df.columns):
        covariates_df = covariates_df.loc[ig.phenotype_df.columns]

    for batch in ig.generate_data():
        if len(batch) == 5:
            P, G_block, v_idx, ids, group_id = batch
            H_block = None
        elif len(batch) == 6:
            P, G_block, v_idx, H_block, ids, group_id = batch
        else:
            raise ValueError("Unexpected grouped batch shape in _run_independent_core_group.")

        # find seed for this group
        seed_rows = seed_by_group_df[seed_by_group_df["group_id"] == group_id]
        if seed_rows.empty:
            continue
        seed_row = seed_rows.iloc[0]
        seed_vid = str(seed_row["variant_id"])

        G_t = torch.tensor(G_block, dtype=torch.float32, device=device)
        H_t = None if H_block is None else torch.tensor(H_block, dtype=torch.float32, device=device)

        # Impute/MAF filter once per window
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
        perms = torch.stack([torch.randperm(n, device=device) for _ in range(nperm)], dim=0)

        dosage_dict = {seed_vid: _dosage_vector_for_covariate(
            genotype_df=ig.genotype_df,
            variant_id=seed_vid,
            sample_order=ig.phenotype_df.columns,
            missing=missing,
        )}
        forward_rows = [seed_row.to_frame().T]

        while True:
            # augmented covariates for this forward step
            if covariates_df is not None:
                C_base = covariates_df.values.astype(np.float32, copy=False)
                C_sel = (np.column_stack([dosage_dict[v] for v in dosage_dict])
                         if dosage_dict else np.zeros((n, 0), np.float32))
                C_aug = np.hstack([C_base, C_sel]).astype(np.float32, copy=False)
            else:
                C_aug = (np.column_stack([dosage_dict[v] for v in dosage_dict]) if dosage_dict
                         else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)
            rez_aug = Residualizer(torch.tensor(C_aug, dtype=torch.float32, device=device)) if C_aug.shape[1] > 0 else None

            # Evaluate all phenotypes in this group; keep the global best
            best = dict(r2=-np.inf, ix_var=-1, ix_pheno=-1, beta=None, se=None, t=None, dof=None)
            r2_perm_list = []

            # For speed, pre-flatten H if present
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

                y_perm = torch.stack([y_resid[idxp] for idxp in perms], dim=1)
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

            # combine permutations across phenotypes (elementwise max)
            r2_perm_max = torch.stack(r2_perm_list, dim=0).max(dim=0).values.detach().cpu().numpy()
            pval_perm = float((np.sum(r2_perm_max >= best["r2"]) + 1) / (r2_perm_max.size + 1))
            pval_beta, a_hat, b_hat = (beta_approx_pval(r2_perm_max, best["r2"]) if beta_approx
                                       else (np.nan, np.nan, np.nan))
            if pval_beta > signif_threshold:
                break

            pid_best = ids[best["ix_pheno"]]
            var_id = variant_df.index.values[v_idx[best["ix_var"]]]
            var_pos = int(variant_df.iloc[v_idx[best["ix_var"]]]["pos"])
            start_pos = ig.phenotype_start[pid_best]
            end_pos = ig.phenotype_end[pid_best]

            row = pd.Series({
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
            forward_rows.append(row.to_frame().T)

            if var_id not in dosage_dict:
                dosage_dict[var_id] = _dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=var_id,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                )

        forward_df = pd.concat(forward_rows, axis=0, ignore_index=True)
        forward_df["rank"] = np.arange(1, forward_df.shape[0] + 1, dtype=int)

        # Backward pass (group)
        if forward_df.shape[0] > 1:
            kept_rows = []
            selected = forward_df["variant_id"].tolist()

            for rk, drop_vid in enumerate(selected, start=1):
                kept = [v for v in selected if v != drop_vid]

                if covariates_df is not None:
                    C_base = covariates_df.values.astype(np.float32, copy=False)
                    C_sel = (np.column_stack([dosage_dict[v] for v in kept]) if kept
                             else np.zeros((n, 0), np.float32))
                    C_aug = np.hstack([C_base, C_sel]).astype(np.float32, copy=False)
                else:
                    C_aug = (np.column_stack([dosage_dict[v] for v in kept]) if kept
                             else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)
                rez_aug = Residualizer(torch.tensor(C_aug, dtype=torch.float32, device=device)) if C_aug.shape[1] > 0 else None

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

                    y_perm = torch.stack([y_resid[idxp] for idxp in perms], dim=1)
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

                if pval_beta <= signif_threshold:
                    pid_best = ids[best["ix_pheno"]]
                    var_id = variant_df.index.values[v_idx[best["ix_var"]]]
                    var_pos = int(variant_df.iloc[v_idx[best["ix_var"]]]["pos"])
                    start_pos = ig.phenotype_start[pid_best]
                    end_pos = ig.phenotype_end[pid_best]

                    row = pd.Series({
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
                    out_rows.append(row.to_frame().T)

        if not forward_rows:
            continue
        if forward_df.shape[0] == 1:
            out_rows.append(forward_df)

    if not out_rows:
        return pd.DataFrame(columns=[
            "group_id","group_size","phenotype_id","variant_id","pos","start_distance","end_distance",
            "num_var","beta","se","tstat","r2_nominal","pval_nominal","pval_perm","pval_beta",
            "beta_shape1","beta_shape2","dof","rank"
        ])
    return pd.concat(out_rows, axis=0, ignore_index=True)


def map_nominal(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        window: int = 1_000_000, nperm: Optional[int] = None, device: str = "cuda",
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

    # Residualize once
    Y = torch.tensor(phenotype_df.values, dtype=torch.float32, device=device)
    with logger.time_block("Residualizing phenotypes", sync=sync):
        Y_resid, rez = _residualize_matrix_with_covariates(Y, covariates_df, device)
        
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
    ig.phenotype_df = pd.DataFrame(
        Y_resid.cpu().numpy(), index=ig.phenotype_df.index, columns=ig.phenotype_df.columns
    )

    # Core compute
    with logger.time_block("Computing associations (nominal)", sync=sync):
        return _run_nominal_core(ig, variant_df, rez, nperm, device,
                                 maf_threshold=maf_threshold)


def map_permutations(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        window: int = 1_000_000, nperm: int = 10_000,
        device: str = "cuda", beta_approx: bool = True,
        logger: SimpleLogger | None = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Empirical cis-QTL mapping (one top variant per phenotype) with permutations.
    Returns a DataFrame with empirical p-values (and optional Beta approximation).
    """
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
    sync = (torch.cuda.synchronize if device == "cuda" else None)

    # Header (tensorQTL-style)
    logger.write("cis-QTL mapping: permutation scan (top per phenotype)")
    logger.write(f"  * device: {device}")
    logger.write(f"  * {phenotype_df.shape[1]} samples")
    logger.write(f"  * {phenotype_df.shape[0]} phenotypes")
    logger.write(f"  * {variant_df.shape[0]} variants")
    logger.write(f"  * cis-window: \u00B1{window:,}")
    logger.write(f"  * nperm={nperm:,} (beta_approx={'on' if beta_approx else 'off'})")
    if maf_threshold and maf_threshold > 0:
        logger.write(f"  * applying in-sample {maf_threshold:g} MAF filter")
    if covariates_df is not None:
        logger.write(f"  * {covariates_df.shape[1]} covariates")
    if haplotypes is not None:
        K = int(haplotypes.shape[2])
        logger.write(f"  * including local ancestry channels (K={K})")

    # Residualize phenotypes once
    Y = torch.tensor(phenotype_df.values, dtype=torch.float32, device=device)
    with logger.time_block("Residualizing phenotypes", sync=sync):
        Y_resid, rez = _residualize_matrix_with_covariates(Y, covariates_df, device)

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
    ig.phenotype_df = pd.DataFrame(Y_resid.cpu().numpy(), index=ig.phenotype_df.index,
                                   columns=ig.phenotype_df.columns)

    # Core either grouped or single-phenotype
    with logger.time_block("Computing associations (permutations)", sync=sync):
        core = _run_permutation_core_group if getattr(ig, "group_s", None) is not None else _run_permutation_core
        return core(ig, variant_df, rez, nperm=nperm, device=device,
                    beta_approx=beta_approx, maf_threshold=maf_threshold)


def map_independent(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, cis_df: pd.DataFrame,
        phenotype_df: pd.DataFrame, phenotype_pos_df: pd.DataFrame,
        covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        fdr: float = 0.05, fdr_col: str = "qval", nperm: int = 10_000,
        window: int = 1_000_000, missing: float = -9.0, random_tiebreak: bool = False,
        device: str = "auto", beta_approx: bool = True,
        logger: SimpleLogger | None = None, verbose: bool = True,
) -> pd.DataFrame:
    """Entry point: build IG; derive seed/threshold from cis_df; dispatch to grouped/ungrouped core."""
    device = ("cuda" if (device in ("auto", None) and torch.cuda.is_available())
              else (device if device in ("cuda", "cpu") else "cpu"))
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
    mt = self.maf_threshold if maf_threshold is None else maf_threshold

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
        logger.write(f"  * including local ancestry channels (K={K})")

    # Build the appropriate input generator (no residualization)
    ig = (
        InputGeneratorCisWithHaps(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
            loci_df=loci_df, group_s=group_s) if haplotypes is not None else
        InputGeneratorCis(
            genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
    )    
    if ig.n_phenotypes == 0:
        raise ValueError("No valid phenotypes after generator preprocessing.")

    # Build seed tables: ungrouped (index by phenotype_id) vs grouped (one per group_id)
    if group_s is None:
        if "phenotype_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'phenotype_id' for ungrouped mapping.")
        signif_seed_df = signif_df.set_index("phenotype_id", drop=False)
        return _run_independent_core(
            ig=ig, variant_df=variant_df, covariates_df=covariates_df,
            signif_seed_df=signif_seed_df, signif_threshold=signif_threshold,
            nperm=nperm, device=device, maf_threshold=maf_threshold,
            random_tiebreak=random_tiebreak, missing=missing, beta_approx=beta_approx,
        )
    else:
        if "group_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'group_id' for grouped mapping.")
        seed_by_group_df = (signif_df.sort_values(["group_id", "pval_beta"])
                                      .groupby("group_id", sort=False).head(1))
        return _run_independent_core_group(
            ig=ig, variant_df=variant_df, covariates_df=covariates_df,
            seed_by_group_df=seed_by_group_df, signif_threshold=signif_threshold,
            nperm=nperm, device=device, maf_threshold=maf_threshold,
            random_tiebreak=random_tiebreak, missing=missing, beta_approx=beta_approx,
        )


class CisMapper:
    """
    Convenience wrapper: build an InputGenerator and run nominal scans.
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
            logger: SimpleLogger | None = None, verbose: bool = True
    ):
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available()) else
                       device if device in ("cuda", "cpu") else "cpu")
        self.logger = logger or SimpleLogger(verbose=verbose, timestamps=True)
        self.variant_df = variant_df
        self.window = window
        self.maf_threshold = maf_threshold

        self.phenotype_df_raw = phenotype_df.copy()                 # keep RAW phenotypes
        self.covariates_df = covariates_df.copy() if covariates_df is not None else None
        self.sample_order = self.ig.phenotype_df.columns.tolist()

        self.ig = (
            InputGeneratorCisWithHaps(
                genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, window=window, haplotypes=haplotypes,
                loci_df=loci_df, group_s=group_s) if haplotypes is not None else
            InputGeneratorCis(
                genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, window=window, group_s=group_s)
        )
        # Residualize phenotypes once
        Y = torch.tensor(self.ig.phenotype_df.values, dtype=torch.float32, device=self.device)
        sync = (torch.cuda.synchronize if self.device == "cuda" else None)
        with self.logger.time_block("Residualizing phenotypes", sync=sync):
            Y_resid, self.rez = _residualize_matrix_with_covariates(Y, covariates_df, self.device)
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

    def map_independent(self, cis_df, fdr=0.05):
        """Forward–backward conditional mapping"""
        pass

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
