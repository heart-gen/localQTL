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
from ..preproc import impute_mean_and_filter, allele_stats, filter_by_maf
from ..regression_kernels import (
    run_batch_regression,
    run_batch_regression_with_permutations
)
from ._permute import make_perm_ix, roll_for_key
from ._buffer import (
    allocate_result_buffers,
    ensure_capacity, write_row,
    buffers_to_dataframe
)
from .common import residualize_matrix_with_covariates, residualize_batch

__all__ = [
    "map_permutations",
]

def _estimate_rows(ig, chrom: str | None, grouped: bool = False) -> int:
    if chrom is None:
        if grouped and getattr(ig, "group_s", None) is not None:
            return int(pd.Index(ig.group_s.dropna().unique()).shape[0])
        return int(ig.phenotype_df.shape[0])
    mask = ig.phenotype_pos_df["chr"] == chrom
    if grouped and getattr(ig, "group_s", None) is not None:
        group_s = ig.group_s.loc[ig.phenotype_pos_df.index]
        return int(pd.Index(group_s[mask].dropna().unique()).shape[0])
    return int(mask.sum())


def _run_permutation_core(
        ig, variant_df, rez, nperm: int, device: str, beta_approx: bool = True,
        maf_threshold: float = 0.0, seed: int | None = None, chrom: str | None = None,
        logger: SimpleLogger | None = None, total_phenotypes: int | None = None,
        perm_ix_t: torch.Tensor | None = None, perm_chunk: int = 4096,
) -> pd.DataFrame:
    """
    One top association per phenotype with empirical permutation p-value (no grouping).
    Compatible with InputGeneratorCis and InputGeneratorCisWithHaps (ungrouped only).
    """
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance", "end_distance",
        "num_var", "slope", "slope_se", "tstat", "r2_nominal", "pval_nominal",
        "pval_perm", "pval_beta", "beta_shape1", "beta_shape2",
        "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof"
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
    }
    buffers = allocate_result_buffers(expected_columns, dtype_map, _estimate_rows(ig, chrom))
    cursor = 0
    processed = 0
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int32, copy=False)

    if total_phenotypes is None:
        total_phenotypes = _estimate_rows(ig, chrom)
    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, total_phenotypes // 10) if total_phenotypes else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"

    for batch in ig.generate_data(chrom=chrom):
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
        y_t = to_device_tensor(p, device, dtype=torch.float32)
        G_t = to_device_tensor(G_block, device, dtype=torch.float32)
        if H_block is None:
            H_t = None
        else:
            H_t = to_device_tensor(H_block, device, dtype=torch.float32)

        # Impute and drop monomorphic
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

        # Minor-allele stats before residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Residualize y/G/H against covariates
        y_resid_t, G_resid, H_resid = residualize_batch(y_t, G_t, H_t, rez, center=True)

        # Compute effective covariate rank for DoF
        k_eff = rez.Q_t.shape[1] if rez is not None else 0

        # Nominal regression on GPU
        betas, ses, tstats = run_batch_regression(
            y=y_resid_t, G=G_resid, H=H_resid, k_eff=k_eff, device=device
        )

        # Partial R^2 for genotype predictor
        n = int(y_resid_t.shape[0])
        p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
        dof = max(n - p_pred, 1)
        ##dof = max(n - 2 - int(k_eff), 1)
        t_g = tstats[:, 0]
        t_sq = t_g.double().pow(2)
        r2_nominal_vec = (t_sq / (t_sq + dof)).to(torch.float32)
        r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
        ix = int(r2_nominal_vec.argmax().item())

        # Extract top variant stats
        beta = float(betas[ix, 0].item())
        se = float(ses[ix, 0].item())
        tval = float(tstats[ix, 0].item())
        r2_nominal = float(r2_nominal_vec[ix].item())

        # Build permutation indices for phenotype
        if perm_ix_t is not None:
            perm_ix_pid = roll_for_key(perm_ix_t, pid, seed)
        else:
            local_seed = subseed(seed, pid) if seed is not None else None
            perm_ix_pid = make_perm_ix(y_resid_t.shape[0], nperm, device, local_seed)

        nperm_local = int(perm_ix_pid.shape[0])
        r2_perm_max = torch.full((nperm_local,), -float("inf"), device=device, dtype=torch.float32)

        for off in range(0, nperm_local, perm_chunk):
            sel = perm_ix_pid[off:off + perm_chunk]
            if sel.numel() == 0:
                continue
            chunk = sel.shape[0]
            y_perm = y_resid_t.index_select(0, sel.reshape(-1)).view(chunk, y_resid_t.shape[0]).transpose(0, 1)
            _, _, _, r2_block = run_batch_regression_with_permutations(
                y=y_resid_t, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
            )
            r2_perm_max[off:off + chunk] = torch.maximum(
                r2_perm_max[off:off + chunk],
                r2_block.to(torch.float32)
            )

        r2_perm_np = r2_perm_max.detach().cpu().numpy()

        # Nominal p (two-sided t)
        pval_nominal = float(get_t_pval(tval, dof))

        # Empirical permutation p (max across variants each perm)
        pval_perm = float((np.sum(r2_perm_np >= r2_nominal) + 1) / (r2_perm_np.size + 1))

        # Optional Beta approximation
        if beta_approx:
            pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                r2_perm_np, r2_nominal, dof_init=dof
            )
        else:
            pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan

        # Metadata
        var_id = idx_to_id[v_idx[ix]]
        var_pos = int(pos_arr[v_idx[ix]])
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = int(var_pos - start_pos)
        end_distance = int(var_pos - end_pos)
        num_var = int(G_resid.shape[0])

        buffers = ensure_capacity(buffers, cursor, 1)
        write_row(
            buffers, cursor,
            {
                "phenotype_id": pid,
                "num_var": num_var,
                "beta_shape1": a_hat,
                "beta_shape2": b_hat,
                "true_dof": true_dof,
                "pval_true_dof": p_true,
                "variant_id": var_id,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "ma_samples": int(ma_samples_t[ix].item()),
                "ma_count": float(ma_count_t[ix].item()),
                "af": float(af_t[ix].item()),
                "slope": beta,
                "slope_se": se,
                "tstat": tval,
                "r2_nominal": r2_nominal,
                "pval_nominal": pval_nominal,
                "pval_perm": pval_perm,
                "pval_beta": pval_beta,
            },
        )
        cursor += 1
        processed += 1
        if (
                logger.verbose
                and total_phenotypes
                and progress_interval
                and (processed % progress_interval == 0 or processed == total_phenotypes)
        ):
            logger.write(
                f"      processed {processed}/{total_phenotypes} phenotypes on {chrom_label}"
            )

    return buffers_to_dataframe(expected_columns, buffers, cursor)


def _run_permutation_core_group(
        ig, variant_df, rez, nperm: int, device: str, beta_approx: bool = True,
        maf_threshold: float = 0.0, seed: int | None = None,
        chrom: str | None = None, logger: SimpleLogger | None = None,
        total_groups: int | None = None, perm_ix_t: torch.Tensor | None = None,
        perm_chunk: int = 4096,
) -> pd.DataFrame:
    """
    Group-aware permutation mapping: returns one top association per *group* (best phenotype within group),
    with empirical p-values computed by taking the max R² across variants and phenotypes for each permutation.
    Mirrors tensorQTL’s grouped behavior.
    """
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    expected_columns = [
        "group_id", "group_size", "phenotype_id", "variant_id", "start_distance",
        "end_distance", "num_var", "slope", "slope_se", "tstat",
        "r2_nominal", "pval_nominal", "pval_perm", "pval_beta", "beta_shape1",
        "beta_shape2", "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof",
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
    }
    buffers = allocate_result_buffers(expected_columns, dtype_map,
                                      _estimate_rows(ig, chrom, grouped=True))
    cursor = 0
    processed = 0
    if total_groups is None:
        total_groups = _estimate_rows(ig, chrom, grouped=True)
    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, total_groups // 10) if total_groups else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"
    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int32, copy=False)
    for batch in ig.generate_data(chrom=chrom):
        # Accept shapes: (P, G, v_idx, ids, group_id) or (P, G, v_idx, H, ids, group_id)
        if len(batch) == 5:
            P, G_block, v_idx, ids, group_id = batch
            H_block = None
        elif len(batch) == 6:
            P, G_block, v_idx, H_block, ids, group_id = batch
        else:
            raise ValueError("Unexpected grouped batch shape.")

        # Tensors for window
        G_t = to_device_tensor(G_block, device, dtype=torch.float32)
        if H_block is None:
            H_t = None
        else:
            H_t = to_device_tensor(H_block, device, dtype=torch.float32)

        # Impute + drop monomorphic
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

        # Minor-allele stats prior to residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Residualize once; reuse G/H residuals for each phenotype
        if isinstance(P, torch.Tensor):
            Y_stack = P.to(device=device, dtype=torch.float32, non_blocking=True)
        elif isinstance(P, (list, tuple)):
            P_arr = np.asarray([np.asarray(pi, dtype=np.float32) for pi in P], dtype=np.float32)
            Y_stack = torch.as_tensor(P_arr, dtype=torch.float32, device=device)
        else:
            P_arr = np.asarray(P, dtype=np.float32)
            if P_arr.ndim == 1:
                P_arr = P_arr[np.newaxis, :]
            Y_stack = torch.as_tensor(P_arr, dtype=torch.float32, device=device)

        if Y_stack.dim() == 1:
            Y_stack = Y_stack.unsqueeze(0)

        # Use shared routine to residualize matrices with the same Residualizer
        mats: list[torch.Tensor] = [G_t]
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
        n = int(Y_resid.shape[1])
        p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
        dof = max(n - p_pred, 1)
        k_eff = rez.Q_t.shape[1] if rez is not None else 0
        ##dof = max(n - 2 - int(k_eff), 1)
        var_ids = idx_to_id[v_idx]
        var_pos = pos_arr[v_idx]

        # Evaluate each phenotype: t-stats -> partial R²; keep the global best (variant, phenotype)
        best = dict(r2=-np.inf, ix_var=-1, ix_pheno=-1, beta=None, se=None, t=None)
        if perm_ix_t is not None:
            perm_ix_group = roll_for_key(perm_ix_t, group_id, seed)
        else:
            local_seed = subseed(seed, group_id) if seed is not None else None
            perm_ix_group = make_perm_ix(n, nperm, device, local_seed)

        nperm_local = int(perm_ix_group.shape[0])
        r2_perm_global_max = torch.full((nperm_local,), -float("inf"), device=device, dtype=torch.float32)

        for j in range(Y_resid.shape[0]):
            y_t = Y_resid[j, :]

            betas, ses, tstats = run_batch_regression(
                y=y_t, G=G_resid, H=H_resid, k_eff=k_eff, device=device
            )
            t_g = tstats[:, 0]
            t_sq = t_g.double().pow(2)
            r2_nominal_vec = (t_sq / (t_sq + dof)).to(torch.float32)
            r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
            ix = int(r2_nominal_vec.argmax().item())
            if float(r2_nominal_vec[ix].item()) > best["r2"]:
                best.update(
                    r2=float(r2_nominal_vec[ix].item()),
                    ix_var=ix,
                    ix_pheno=j,
                    beta=float(betas[ix, 0].item()),
                    se=float(ses[ix, 0].item()),
                    t=float(tstats[ix, 0].item()),
                )

            for off in range(0, nperm_local, perm_chunk):
                sel = perm_ix_group[off:off + perm_chunk]
                if sel.numel() == 0:
                    continue
                chunk = sel.shape[0]
                y_perm = y_t.index_select(0, sel.reshape(-1)).view(chunk, y_t.shape[0]).transpose(0, 1)
                _, _, _, r2_block = run_batch_regression_with_permutations(
                    y=y_t, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff, device=device
                )
                r2_perm_global_max[off:off + chunk] = torch.maximum(
                    r2_perm_global_max[off:off + chunk],
                    r2_block.to(torch.float32)
                )

        r2_perm_max = r2_perm_global_max.detach().cpu().numpy()

        # Build output (metadata for the winning phenotype/variant)
        pid = ids[best["ix_pheno"]]
        var_id = var_ids[best["ix_var"]]
        pos = int(var_pos[best["ix_var"]])
        start_pos = ig.phenotype_start[pid]
        end_pos = ig.phenotype_end[pid]
        start_distance = int(pos - start_pos)
        end_distance = int(pos - end_pos)
        num_var = int(G_resid.shape[0])

        # p-values
        pval_nominal = float(get_t_pval(best["t"], dof))
        pval_perm = float((np.sum(r2_perm_max >= best["r2"]) + 1) / (r2_perm_max.size + 1))
        if beta_approx:
            pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                r2_perm_max, best["r2"], dof_init=dof
            )
        else:
            pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan

        buffers = ensure_capacity(buffers, cursor, 1)
        write_row(
            buffers, cursor,
            {
                "group_id": group_id,
                "group_size": len(ids),
                "phenotype_id": pid,
                "variant_id": var_id,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "num_var": num_var,
                "slope": best["beta"],
                "slope_se": best["se"],
                "tstat": best["t"],
                "r2_nominal": best["r2"],
                "pval_nominal": pval_nominal,
                "pval_perm": pval_perm,
                "pval_beta": pval_beta,
                "beta_shape1": a_hat,
                "beta_shape2": b_hat,
                "ma_samples": int(ma_samples_t[best["ix_var"]].item()),
                "ma_count": float(ma_count_t[best["ix_var"]].item()),
                "af": float(af_t[best["ix_var"]].item()),
                "true_dof": true_dof,
                "pval_true_dof": p_true,
            },
        )
        cursor += 1
        processed += 1
        if (
                logger.verbose
                and total_groups
                and progress_interval
                and (processed % progress_interval == 0 or processed == total_groups)
        ):
            logger.write(
                f"      processed {processed}/{total_groups} groups on {chrom_label}"
            )

    return buffers_to_dataframe(expected_columns, buffers, cursor)


def map_permutations(
        genotype_df: pd.DataFrame, variant_df: pd.DataFrame, phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame, covariates_df: Optional[pd.DataFrame] = None,
        haplotypes: Optional[object] = None, loci_df: Optional[pd.DataFrame] = None,
        group_s: Optional[pd.Series] = None, maf_threshold: float = 0.0,
        window: int = 1_000_000, nperm: int = 10_000, device: str = "cuda",
        perm_chunk: int = 4096, beta_approx: bool = True, seed: int | None = None,
        logger: SimpleLogger | None = None, verbose: bool = True,
        preload_haplotypes: bool = True, tensorqtl_flavor: bool = False,
) -> pd.DataFrame:
    """
    Empirical cis-QTL mapping (one top variant per phenotype) with permutations.
    Returns a DataFrame with empirical p-values (and optional Beta approximation).

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
    logger.write("cis-QTL mapping: permutation scan (top per phenotype)")
    logger.write(f"  * device: {device}")
    logger.write(f"  * {phenotype_df.shape[1]} samples")
    logger.write(f"  * {phenotype_df.shape[0]} phenotypes")
    logger.write(f"  * {variant_df.shape[0]} variants")
    logger.write(f"  * cis-window: \u00B1{window:,}")
    logger.write(f"  * nperm={nperm:,} (beta_approx={'on' if beta_approx else 'off'})")
    logger.write(f"  * seed: {seed}")
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

    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

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

    # Residualize phenotypes once (after generator filters constants/missing)
    Y = to_device_tensor(ig.phenotype_df.values, device, dtype=torch.float32)
    with logger.time_block("Residualizing phenotypes", sync=sync):
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_df,
                                                          device, tensorqtl_flavor)

    ig.phenotype_df = pd.DataFrame(Y_resid.cpu().numpy(), index=ig.phenotype_df.index,
                                   columns=ig.phenotype_df.columns)

    n_samples = int(ig.phenotype_df.shape[1])
    perm_ix_t = make_perm_ix(n_samples, nperm, device, seed)

    # Core either grouped or single-phenotype
    phenotype_counts = ig.phenotype_pos_df['chr'].value_counts().to_dict()
    total_phenotypes = int(ig.phenotype_df.shape[0])
    if logger.verbose:
        logger.write(f"    Mapping all chromosomes ({total_phenotypes} phenotypes)")

    group_mode = getattr(ig, "group_s", None) is not None
    core = _run_permutation_core_group if group_mode else _run_permutation_core

    overall_start = time.time()
    results: list[pd.DataFrame] = []
    with logger.time_block("Permutation scan (per-chrom)", sync=sync, sec=False):
        for chrom in ig.chrs:
            chrom_total = int(phenotype_counts.get(chrom, 0))
            if logger.verbose:
                logger.write(f"    Mapping chromosome {chrom} ({chrom_total} phenotypes)")
            chrom_start = time.time()
            with logger.time_block(f"{chrom}: map_permutations", sync=sync):
                total_units = _estimate_rows(ig, chrom, grouped=group_mode)
                if group_mode:
                    chrom_df = core(
                        ig, variant_df, rez, nperm=nperm, device=device,
                        beta_approx=beta_approx, maf_threshold=maf_threshold,
                        seed=seed, chrom=chrom,
                        logger=logger, total_groups=total_units,
                        perm_ix_t=perm_ix_t, perm_chunk=perm_chunk,
                    )
                else:
                    chrom_df = core(
                        ig, variant_df, rez, nperm=nperm, device=device,
                        beta_approx=beta_approx, maf_threshold=maf_threshold,
                        seed=seed, chrom=chrom,
                        logger=logger, total_phenotypes=total_units,
                        perm_ix_t=perm_ix_t, perm_chunk=perm_chunk,
                    )
            results.append(chrom_df)
            if logger.verbose:
                elapsed = time.time() - chrom_start
                logger.write(f"    Chromosome {chrom} completed in {elapsed:.2f}s")

    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed permutation scan in {elapsed:.2f}s")

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    return pd.DataFrame()
