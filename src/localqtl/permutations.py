import torch
import numpy as np
import pandas as pd
from typing import Optional, List

from ..utils import SimpleLogger
from ..stats import beta_approx_pval, get_t_pval
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, allele_stats, filter_by_maf
from ..regression_kernels import (
    run_batch_regression,
    run_batch_regression_with_permutations
)
from .common import residualize_matrix_with_covariates, residualize_batch

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
        y_t, G_t, H_t = residualize_batch(y_t, G_t, H_t, rez, center=True)

        # Compute effective covariate rank for DoF
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Build permuted phenotypes (n x nperm)
        perms = [torch.randperm(y_t.shape[0], device=device) for _ in range(nperm)]
        y_perm = torch.stack([y_t[idxp] for idxp in perms], dim=1)
        
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
            y_perm = torch.stack([y_t[idxp] for idxp in perms], dim=1)  # (n x nperm)

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
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_df, device)

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
