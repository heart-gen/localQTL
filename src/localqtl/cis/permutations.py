import torch, time
import numpy as np
import pandas as pd
from typing import Optional

try:
    torch.backends.fp32_precision = "ieee"
except Exception:
    pass

from ..utils import SimpleLogger, subseed
from .._torch_utils import to_device_tensor
from ..stats import beta_approx_pval, get_f_pval, get_t_pval, _beta_mle_on_p
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import (impute_mean_and_filter, allele_stats, filter_by_maf,
                       ancestry_allele_stats)
from ..regression_kernels import (
    run_batch_regression,
    run_batch_dosage_regression,
    prep_ctx_for_dosage_perm,
    dosage_fstat,
    dosage_perm_chunk_fmax,
)
from ._permute import PermutationStream, compute_perm_r2_max
from ._buffer import (
    allocate_result_buffers,
    ensure_capacity, write_row,
    buffers_to_dataframe
)
from .nominal import _paired_covariate_context
from .common import (
    auto_perm_chunk,
    residualize_matrix_with_covariates,
    validate_ancestry_model,
    project_interaction,
    residualize_batch,
    residualize_batch_dosage,
    residualize_batch_interaction,
    prepare_interaction_covariate,
)
from .nominal import _dosage_columns, _n_ancestries, dosage_result_arrays

__all__ = [
    "map_permutations",
]

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

def _nanmax(x: torch.Tensor, dim: int):
    if hasattr(torch, "nanmax"):
        return torch.nanmax(x, dim=dim)
    replace = torch.full_like(x, float("-inf"))
    return torch.max(torch.where(torch.isnan(x), replace, x), dim=dim)

def _dosage_lead_row(y_resid, D_resid, L_resid, G_t, H_t, ix: int, k_eff: int,
                     min_ancestry_ac: float, device: str):
    """
    Dosage columns of the lead variant ``ix``, refitted on its own.

    Only the lead is refitted: running ``run_batch_dosage_regression`` over
    the whole window would solve one ``p x p`` system per variant and build
    ``(m, K)`` and ``(m, K, K)`` tensors just so that a single row can be read
    out of them.  Its ``fstat`` is also the numerically stable estimate of the
    joint F -- ``_dosage_fstat_block`` forms ``y'y - y'XB`` in float32, which
    cancels catastrophically once the fit is near-perfect -- so it, and not
    the block statistic, is what the row reports as ``fstat_joint`` and
    ``pval_nominal``.  The per-ancestry columns come from the same
    :func:`localqtl.cis.nominal.dosage_result_arrays` the nominal scan uses,
    so the two outputs cannot drift apart.

    Returns ``(dosage_vals, slope, slope_se, tstat, dof)``, the last four
    summarising the lead by its strongest unmasked ancestry.
    """
    one = slice(ix, ix + 1)
    anc_stats = {k: v.detach().cpu().numpy().astype(np.float32)
                 for k, v in ancestry_allele_stats(G_t[one], H_t[one]).items()}
    res = run_batch_dosage_regression(
        y=y_resid, D=D_resid[one], L=None if L_resid is None else L_resid[one],
        k_eff=k_eff, device=device,
    )
    vals, slope, slope_se, tstat, _, dof = dosage_result_arrays(
        res, anc_stats=anc_stats, min_ancestry_ac=min_ancestry_ac
    )
    dosage_vals = {c: float(np.asarray(v).reshape(-1)[0]) for c, v in vals.items()}
    return dosage_vals, float(slope[0]), float(slope_se[0]), float(tstat[0]), int(dof)


def _beta_fit_on_minp(minp_null: np.ndarray, p_obs: float) -> tuple[float, float, float]:
    """
    Beta approximation of the per-phenotype minimum p-value (FastQTL's
    argument applied directly to p-values, as the joint dosage test is an
    F test rather than a correlation).  Returns ``(pval_beta, shape1, shape2)``.
    """
    from scipy import stats as _st
    a, b = _beta_mle_on_p(np.asarray(minp_null, dtype=np.float64))
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan"), float("nan"), float("nan")
    return float(_st.beta.cdf(p_obs, a, b)), float(a), float(b)


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
        perm_chunk: int = 4096, perm_indices_mode: str = "generator",
        mixed_precision: str | None = None,
        ancestry_model: str | None = None,
        n_ancestries: int | None = None,
        interaction_covariate_t: torch.Tensor | None = None,
        paired_ctx: dict | None = None,
        min_ancestry_ac: float = 0.0,
) -> pd.DataFrame:
    """
    One top association per phenotype with empirical permutation p-value (no grouping).
    ``paired_ctx`` maps phenotype IDs to ``(Residualizer, residualized y)`` for
    phenotype-specific covariates (see ``map_nominal``).
    Compatible with InputGeneratorCis and InputGeneratorCisWithHaps (ungrouped only).
    """
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance", "end_distance",
        "num_var", "slope", "slope_se", "tstat", "r2_nominal", "pval_nominal",
        "pval_perm", "pval_beta", "beta_shape1", "beta_shape2",
        "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof"
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
    dosage_columns: list[str] = []
    if ancestry_model == "dosage" and n_ancestries is not None and n_ancestries > 0:
        dosage_columns = _dosage_columns(int(n_ancestries))
        expected_columns.extend(dosage_columns)
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
    for col in interaction_columns + dosage_columns:
        dtype_map[col] = np.float32
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

    perm_indices_mode = (perm_indices_mode or "generator").lower()
    if perm_indices_mode not in {"generator", "cpu_pinned", "gpu"}:
        perm_indices_mode = "generator"

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

        # Phenotype-specific (paired) covariates: swap in that phenotype's residualizer
        rez_b = rez
        if paired_ctx and pid in paired_ctx:
            rez_b, p = paired_ctx[pid]

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

        # Sanity checks to catch any future drift
        assert G_t.shape[0] == v_idx.shape[0], "G_t and v_idx out of sync"
        if H_t is not None:
            assert H_t.shape[0] == G_t.shape[0], "G_t and H_t out of sync"

        # Minor-allele stats before residualization
        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

        # Residualize y/G/H against covariates
        interaction_mode = ancestry_model == "interaction" and H_t is not None
        dosage_mode = ancestry_model == "dosage" and H_t is not None
        covar_interaction = interaction_covariate_t is not None
        if dosage_mode:
            y_resid_t, D_resid, L_resid = residualize_batch_dosage(
                y_t, G_t, H_t, rez_b, center=True, group=False
            )
            G_resid, H_resid = G_t, None
        elif interaction_mode:
            y_resid_t, G_resid, H_resid, I_resid = residualize_batch_interaction(
                y_t, G_t, H_t, rez_b, center=True, group=False
            )
        else:
            y_resid_t, G_resid, H_resid = residualize_batch(y_t, G_t, H_t, rez_b, center=True)

        # Compute effective covariate rank for DoF
        k_eff = rez_b.k_eff if rez_b is not None else 1

        dosage_vals: dict[str, float] | None = None
        if dosage_mode:
            # Joint Wald F over the K ancestry effects is the per-variant
            # statistic; the permutation statistic is its maximum over the
            # window (equivalently the smallest joint p-value).
            K = int(D_resid.shape[2])
            ctx = prep_ctx_for_dosage_perm(D_resid, L_resid, k_eff=k_eff)
            dof = int(ctx["dof"])
            f_obs = torch.nan_to_num(dosage_fstat(ctx, y_resid_t), nan=-1.0)
            ix = int(f_obs.argmax().item())
            # Two estimates of the lead's F, deliberately: ``f_block`` shares
            # the estimator (and so the float32 error) of the permutation
            # null and is what the null is compared against, while the refit
            # below gives the stable value the row reports.
            f_block = float(f_obs[ix].item())

            dosage_vals, beta, se, tval, _ = _dosage_lead_row(
                y_resid_t, D_resid, L_resid, G_t, H_t, ix, k_eff, min_ancestry_ac, device
            )
            f_lead = float(dosage_vals["fstat_joint"])
            pval_nominal = float(dosage_vals["pval_joint"])
            r2_nominal = float(K * f_lead / (K * f_lead + dof)) if f_lead > 0 else 0.0

            n_samples = int(y_resid_t.shape[0])
            perm_chunk_local = auto_perm_chunk(int(D_resid.shape[0]), nperm,
                                               pH=int(ctx["p"]))
            stream = PermutationStream(
                n_samples=n_samples,
                nperm=nperm,
                device=y_resid_t.device,
                chunk_size=max(1, min(int(perm_chunk), perm_chunk_local)),
                seed=subseed(seed, pid) if seed is not None else None,
                mode=perm_indices_mode,
            )
            fmax_parts = []
            for sel in stream.iter_chunks(nperm):
                flat = sel.reshape(-1)
                y_perm = y_resid_t.index_select(0, flat).view(sel.shape[0], n_samples).transpose(0, 1)
                fmax_parts.append(dosage_perm_chunk_fmax(ctx, y_perm))
            fmax_np = (torch.cat(fmax_parts).detach().cpu().numpy() if fmax_parts
                       else np.empty(0, dtype=np.float64))
            pval_perm = float((np.sum(fmax_np >= f_block) + 1) / (fmax_np.size + 1))
            if beta_approx and fmax_np.size:
                minp_null = get_f_pval(fmax_np, K, dof)
                pval_beta, a_hat, b_hat = _beta_fit_on_minp(
                    minp_null, float(get_f_pval(f_block, K, dof)))
            else:
                pval_beta = a_hat = b_hat = np.nan
            # The joint test is an F test, so there is no t-style effective
            # dof to re-estimate: report the residual dof and leave p_true NaN.
            true_dof = dof
            p_true = np.nan
        elif interaction_mode:
            if H_resid is None:
                raise ValueError("interaction model requires haplotypes.")
            pH = int(H_resid.shape[2])
            n_anc = int(n_ancestries) if n_ancestries is not None else pH
            H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid], dim=2)
            n = int(y_resid_t.shape[0])
            p_pred = 1 + int(H_cov.shape[2])
            dof = max(n - int(k_eff) - int(p_pred), 1)

            anc_results = []
            r2_perm_global = None
            best_r2_val = -float("inf")
            best_ix_var = -1
            best_anc = -1

            perm_seed_base = subseed(seed, pid) if seed is not None else None

            for k in range(pH):
                stream = PermutationStream(
                    n_samples=int(y_resid_t.shape[0]),
                    nperm=nperm,
                    device=y_resid_t.device,
                    chunk_size=max(1, int(perm_chunk)),
                    seed=perm_seed_base,
                    mode=perm_indices_mode,
                )
                betas_k, ses_k, tstats_k, r2_nominal_vec, r2_perm_vec = compute_perm_r2_max(
                    y_resid=y_resid_t,
                    G_resid=I_resid[:, :, k],
                    H_resid=H_cov,
                    k_eff=k_eff,
                    perm_stream=stream,
                    max_permutations=nperm,
                    return_nominal=True,
                    mixed_precision=mixed_precision,
                )
                if r2_nominal_vec is None:
                    r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid_t.device, dtype=torch.float32)
                r2_nominal_vec = torch.nan_to_num(r2_nominal_vec.to(torch.float32), nan=-1.0)
                r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                ix = int(ix_t.item())
                anc_results.append((betas_k, ses_k, tstats_k))

                if r2_max_t > best_r2_val:
                    best_r2_val = float(r2_max_t.item())
                    best_ix_var = ix
                    best_anc = k

                r2_perm_global = r2_perm_vec if r2_perm_global is None else torch.maximum(r2_perm_global, r2_perm_vec)

            if best_ix_var < 0:
                continue

            slope_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
            slope_se_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
            tstat_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
            pval_gxh = np.full((n_anc,), np.nan, dtype=np.float32)

            for k in range(pH):
                betas_k, ses_k, tstats_k = anc_results[k]
                slope_gxh[k] = float(betas_k[best_ix_var, 0].item())
                slope_se_gxh[k] = float(ses_k[best_ix_var, 0].item())
                tstat_gxh[k] = float(tstats_k[best_ix_var, 0].item())
                pval_gxh[k] = float(get_t_pval(tstat_gxh[k], dof))

            beta = float(slope_gxh[best_anc])
            se = float(slope_se_gxh[best_anc])
            tval = float(tstat_gxh[best_anc])
            r2_nominal = float(best_r2_val)
            ix = int(best_ix_var)

            r2_perm_np = r2_perm_global.detach().cpu().numpy()
            pval_nominal = float(get_t_pval(tval, dof))
            pval_perm = float((np.sum(r2_perm_np >= r2_nominal) + 1) / (r2_perm_np.size + 1))

            if beta_approx:
                pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                    r2_perm_np, r2_nominal, dof_init=dof
                )
            else:
                pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
        elif covar_interaction:
            c_t = interaction_covariate_t
            if c_t.device != G_resid.device:
                c_t = c_t.to(G_resid.device)
            # g:c from the raw genotypes and c itself are projected like g (Frisch–Waugh)
            I_t = project_interaction(G_t * c_t, rez_b)
            c_res = project_interaction(c_t.unsqueeze(0), rez_b)[0]
            c_rep = c_res.unsqueeze(0).expand(G_resid.shape[0], -1).unsqueeze(-1)
            if H_resid is None:
                H_cov = torch.cat([G_resid.unsqueeze(-1), c_rep], dim=2)
            else:
                H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid, c_rep], dim=2)

            n = int(y_resid_t.shape[0])
            p_pred = 1 + int(H_cov.shape[2])
            dof = max(n - int(k_eff) - int(p_pred), 1)

            betas, ses, tstats, r2_nominal_vec, r2_perm_max = compute_perm_r2_max(
                y_resid=y_resid_t,
                G_resid=I_t,
                H_resid=H_cov,
                k_eff=k_eff,
                perm_stream=PermutationStream(
                    n_samples=int(y_resid_t.shape[0]),
                    nperm=nperm,
                    device=y_resid_t.device,
                    chunk_size=max(1, int(perm_chunk)),
                    seed=subseed(seed, pid) if seed is not None else None,
                    mode=perm_indices_mode,
                ),
                max_permutations=nperm,
                return_nominal=True,
                mixed_precision=mixed_precision,
            )
            if r2_nominal_vec is None:
                r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid_t.device, dtype=torch.float32)
            r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
            ix = int(r2_nominal_vec.argmax().item())

            beta = float(betas[ix, 0].item())
            se = float(ses[ix, 0].item())
            tval = float(tstats[ix, 0].item())
            t_sq = float(tval * tval)
            r2_nominal = float(t_sq / (t_sq + dof))

            r2_perm_np = r2_perm_max.detach().cpu().numpy()
            pval_nominal = float(get_t_pval(tval, dof))
            pval_perm = float((np.sum(r2_perm_np >= r2_nominal) + 1) / (r2_perm_np.size + 1))

            if beta_approx:
                pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                    r2_perm_np, r2_nominal, dof_init=dof
                )
            else:
                pval_beta = a_hat = b_hat = true_dof = p_true =  np.nan
        else:
            # Nominal regression on GPU
            betas, ses, tstats = run_batch_regression(
                y=y_resid_t, G=G_resid, H=H_resid, k_eff=k_eff, device=device
            )

            # Partial R^2 for genotype predictor
            n = int(y_resid_t.shape[0])
            p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
            dof = max(n - int(k_eff) - p_pred, 1)
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
            perm_stream = PermutationStream(
                n_samples=int(y_resid_t.shape[0]),
                nperm=nperm,
                device=y_resid_t.device,
                chunk_size=max(1, int(perm_chunk)),
                seed=subseed(seed, pid) if seed is not None else None,
                mode=perm_indices_mode,
            )
            _, _, _, _, r2_perm_max = compute_perm_r2_max(
                y_resid=y_resid_t,
                G_resid=G_resid,
                H_resid=H_resid,
                k_eff=k_eff,
                perm_stream=perm_stream,
                max_permutations=nperm,
                return_nominal=False,
                mixed_precision=mixed_precision,
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
        row = {
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
        }
        if interaction_mode and interaction_columns:
            for anc_idx in range(len(interaction_columns) // 4):
                base = anc_idx * 4
                row[interaction_columns[base + 0]] = slope_gxh[anc_idx]
                row[interaction_columns[base + 1]] = slope_se_gxh[anc_idx]
                row[interaction_columns[base + 2]] = tstat_gxh[anc_idx]
                row[interaction_columns[base + 3]] = pval_gxh[anc_idx]
        if dosage_mode and dosage_vals is not None:
            for col in dosage_columns:
                row[col] = dosage_vals.get(col, np.nan)
        write_row(buffers, cursor, row)
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
        total_groups: int | None = None, perm_chunk: int = 4096,
        perm_indices_mode: str = "generator", mixed_precision: str | None = None,
        ancestry_model: str | None = None,
        n_ancestries: int | None = None,
        interaction_covariate_t: torch.Tensor | None = None,
        min_ancestry_ac: float = 0.0,
) -> pd.DataFrame:
    """
    Group-aware permutation mapping: returns one top association per group with empirical p-values.
    """
    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")

    perm_indices_mode = (perm_indices_mode or "generator").lower()
    if perm_indices_mode not in {"generator", "cpu_pinned", "gpu"}:
        perm_indices_mode = "generator"

    expected_columns = [
        "group_id", "group_size", "phenotype_id", "variant_id", "start_distance",
        "end_distance", "num_var", "slope", "slope_se", "tstat",
        "r2_nominal", "pval_nominal", "pval_perm", "pval_beta", "beta_shape1",
        "beta_shape2", "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof",
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
    dosage_columns: list[str] = []
    if ancestry_model == "dosage" and n_ancestries is not None and n_ancestries > 0:
        dosage_columns = _dosage_columns(int(n_ancestries))
        expected_columns.extend(dosage_columns)
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
    for col in interaction_columns + dosage_columns:
        dtype_map[col] = np.float32
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
        if len(batch) == 5:
            P, G_block, v_idx, ids, group_id = batch
            H_block = None
        elif len(batch) == 6:
            P, G_block, v_idx, H_block, ids, group_id = batch
        else:
            raise ValueError("Unexpected grouped batch shape.")

        G_t = to_device_tensor(G_block, device, dtype=torch.float32)
        H_t = None if H_block is None else to_device_tensor(H_block, device, dtype=torch.float32)

        G_t, keep_mono, _ = impute_mean_and_filter(G_t)
        if G_t.shape[0] == 0:
            continue

        v_idx = v_idx[keep_mono.detach().cpu().numpy()]
        if H_t is not None:
            H_t = H_t[keep_mono]
            # The dosage model needs every ancestry channel (sum_k D_k = G);
            # the other models drop the last one for identifiability.
            if H_t.shape[2] > 1 and ancestry_model != "dosage":
                H_t = H_t[:, :, :-1]

        if maf_threshold and maf_threshold > 0:
            keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
            if keep_maf.sum().item() == 0:
                continue
            v_idx = v_idx[keep_maf.detach().cpu().numpy()]
            G_t = G_t[keep_maf]
            if H_t is not None:
                H_t = H_t[keep_maf]

        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)

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

        interaction_mode = ancestry_model == "interaction" and H_t is not None
        dosage_mode = ancestry_model == "dosage" and H_t is not None
        covar_interaction = interaction_covariate_t is not None
        D_resid = L_resid = None
        if dosage_mode:
            Y_resid, D_resid, L_resid = residualize_batch_dosage(
                Y_stack, G_t, H_t, rez, center=True, group=True
            )
            G_resid, H_resid = G_t, None
        elif interaction_mode:
            Y_resid, G_resid, H_resid, I_resid = residualize_batch_interaction(
                Y_stack, G_t, H_t, rez, center=True, group=True
            )
        else:
            Y_resid, G_resid, H_resid = residualize_batch(
                Y_stack, G_t, H_t, rez, center=True, group=True
            )

        ids_list = list(ids)
        perm_seed_base = subseed(seed, f"group:{group_id}") if seed is not None else None

        best_ix_var = -1
        best_ix_pheno = -1
        best_beta = best_se = best_t = None
        best_dof = 0
        best_r2_val = -np.inf
        slope_gxh = slope_se_gxh = tstat_gxh = pval_gxh = None
        pval_perm = float("nan")
        pval_beta = float("nan")
        a_hat = float("nan")
        b_hat = float("nan")
        p_true = float("nan")
        true_dof = float("nan")
        pval_nominal = float("nan")
        dosage_vals: dict[str, float] | None = None
        num_var = int(G_resid.shape[0])

        r2_perm_global = torch.full((nperm,), -float("inf"), device=G_resid.device, dtype=torch.float32)

        if dosage_mode:
            # The joint Wald F over the K ancestry effects is the per-variant
            # statistic; the permutation statistic is its maximum over the
            # window *and* over the phenotypes of the group.  Every phenotype
            # of a group shares samples, covariates, window and K, so the
            # residual dof is identical and the largest F is the smallest
            # joint p-value.  ``ctx`` depends only on the (shared) residualized
            # design, so it is built once and reused for every phenotype and
            # every permutation chunk.
            K = int(D_resid.shape[2])
            k_eff_g = rez.k_eff if rez is not None else 1
            ctx = prep_ctx_for_dosage_perm(D_resid, L_resid, k_eff=k_eff_g)
            dof = int(ctx["dof"])
            n_samples_g = int(D_resid.shape[1])
            best_f_val = -float("inf")
            f_perm_global = None
            perm_chunk_g = max(1, min(int(perm_chunk),
                                      auto_perm_chunk(int(D_resid.shape[0]), nperm,
                                                      pH=int(ctx["p"]))))

            for j, y_resid in enumerate(Y_resid):
                f_obs = torch.nan_to_num(dosage_fstat(ctx, y_resid), nan=-1.0)
                f_max_t, ix_t = torch.max(f_obs, dim=0)
                f_max = float(f_max_t.item())
                if f_max > best_f_val:
                    best_f_val = f_max
                    best_ix_var = int(ix_t.item())
                    best_ix_pheno = int(j)
                # Every phenotype of the group is permuted with the same seed,
                # so the maximum across the group preserves the within-group
                # correlation structure (as in the correlation-based path).
                stream = PermutationStream(
                    n_samples=n_samples_g,
                    nperm=nperm,
                    device=y_resid.device,
                    chunk_size=perm_chunk_g,
                    seed=perm_seed_base,
                    mode=perm_indices_mode,
                )
                fmax_parts = []
                for sel in stream.iter_chunks(nperm):
                    flat = sel.reshape(-1)
                    y_perm = y_resid.index_select(0, flat).view(
                        sel.shape[0], n_samples_g).transpose(0, 1)
                    fmax_parts.append(dosage_perm_chunk_fmax(ctx, y_perm))
                if fmax_parts:
                    f_perm_pheno = torch.cat(fmax_parts)
                    f_perm_global = (f_perm_pheno if f_perm_global is None
                                     else torch.maximum(f_perm_global, f_perm_pheno))

            if best_ix_var >= 0:
                best_dof = dof
                fmax_np = (f_perm_global.detach().cpu().numpy()
                           if f_perm_global is not None else np.empty(0, dtype=np.float64))
                # ``best_f_val`` shares the estimator of the null, so it is
                # what the null is compared against; the refit below gives the
                # stable F that the row reports.
                pval_perm = float((np.sum(fmax_np >= best_f_val) + 1) / (fmax_np.size + 1))
                if beta_approx and fmax_np.size:
                    pval_beta, a_hat, b_hat = _beta_fit_on_minp(
                        get_f_pval(fmax_np, K, dof), float(get_f_pval(best_f_val, K, dof))
                    )
                else:
                    pval_beta = a_hat = b_hat = np.nan
                # The joint test is an F test, so there is no t-style effective
                # dof to re-estimate: report the residual dof and leave p_true NaN.
                true_dof = dof
                p_true = np.nan

                dosage_vals, best_beta, best_se, best_t, _ = _dosage_lead_row(
                    Y_resid[best_ix_pheno], D_resid, L_resid, G_t, H_t,
                    best_ix_var, k_eff_g, min_ancestry_ac, device,
                )
                f_lead_g = float(dosage_vals["fstat_joint"])
                pval_nominal = float(dosage_vals["pval_joint"])
                best_r2_val = (float(K * f_lead_g / (K * f_lead_g + dof))
                               if f_lead_g > 0 else 0.0)
        else:
            for j, (pid_inner, y_resid) in enumerate(zip(ids_list, Y_resid)):
                if interaction_mode:
                    if H_resid is None:
                        raise ValueError("interaction model requires haplotypes.")
                    pH = int(H_resid.shape[2])
                    n_anc = int(n_ancestries) if n_ancestries is not None else pH
                    H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid], dim=2)
                    k_eff = rez.k_eff if rez is not None else 1
                    p_pred = 1 + int(H_cov.shape[2])
                    dof = max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1)

                    anc_results = []
                    r2_perm_pheno = None
                    best_r2_pheno = -float("inf")
                    best_ix_var_pheno = -1
                    best_anc_pheno = -1

                    for k in range(pH):
                        stream = PermutationStream(
                            n_samples=int(y_resid.shape[0]),
                            nperm=nperm,
                            device=y_resid.device,
                            chunk_size=max(1, int(perm_chunk)),
                            seed=perm_seed_base,
                            mode=perm_indices_mode,
                        )
                        betas, ses, tstats, r2_nominal_vec, r2_perm_vec = compute_perm_r2_max(
                            y_resid=y_resid,
                            G_resid=I_resid[:, :, k],
                            H_resid=H_cov,
                            k_eff=k_eff,
                            perm_stream=stream,
                            max_permutations=nperm,
                            return_nominal=True,
                            mixed_precision=mixed_precision,
                        )
                        if r2_nominal_vec is None:
                            r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid.device, dtype=torch.float32)
                        r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
                        r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                        ix = int(ix_t.item())
                        anc_results.append((betas, ses, tstats))

                        if r2_max_t > best_r2_pheno:
                            best_r2_pheno = float(r2_max_t.item())
                            best_ix_var_pheno = ix
                            best_anc_pheno = k

                        r2_perm_pheno = r2_perm_vec if r2_perm_pheno is None else torch.maximum(r2_perm_pheno, r2_perm_vec)

                    if best_ix_var_pheno >= 0 and best_r2_pheno > best_r2_val:
                        best_r2_val = float(best_r2_pheno)
                        best_ix_var = int(best_ix_var_pheno)
                        best_ix_pheno = int(j)
                        best_dof = int(dof)
                        # Store per-ancestry stats for this phenotype
                        slope_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        slope_se_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        tstat_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        pval_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        for k in range(pH):
                            betas, ses, tstats = anc_results[k]
                            slope_gxh[k] = float(betas[best_ix_var, 0].item())
                            slope_se_gxh[k] = float(ses[best_ix_var, 0].item())
                            tstat_gxh[k] = float(tstats[best_ix_var, 0].item())
                            pval_gxh[k] = float(get_t_pval(tstat_gxh[k], dof))
                        best_beta = float(slope_gxh[best_anc_pheno])
                        best_se = float(slope_se_gxh[best_anc_pheno])
                        best_t = float(tstat_gxh[best_anc_pheno])

                    r2_perm_global = torch.maximum(r2_perm_global, r2_perm_pheno)
                elif covar_interaction:
                    c_t = interaction_covariate_t
                    if c_t.device != G_resid.device:
                        c_t = c_t.to(G_resid.device)
                    I_t = project_interaction(G_t * c_t, rez)
                    c_res = project_interaction(c_t.unsqueeze(0), rez)[0]
                    c_rep = c_res.unsqueeze(0).expand(G_resid.shape[0], -1).unsqueeze(-1)
                    if H_resid is None:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), c_rep], dim=2)
                    else:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid, c_rep], dim=2)
                    k_eff = rez.k_eff if rez is not None else 1
                    p_pred = 1 + int(H_cov.shape[2])
                    dof = max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1)

                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, int(perm_chunk)),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    betas, ses, tstats, r2_nominal_vec, r2_perm_vec = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=I_t,
                        H_resid=H_cov,
                        k_eff=k_eff,
                        perm_stream=stream,
                        max_permutations=nperm,
                        return_nominal=True,
                        mixed_precision=mixed_precision,
                    )
                    if r2_nominal_vec is None:
                        r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid.device, dtype=torch.float32)
                    r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    t_g = tstats[:, 0].double()
                    if r2_max_t > best_r2_val:
                        best_r2_val = float(r2_max_t.item())
                        best_ix_var = ix
                        best_ix_pheno = j
                        best_beta = float(betas[ix, 0].item())
                        best_se = float(ses[ix, 0].item())
                        best_t = float(t_g[ix].item())
                        best_dof = int(dof)
                    r2_perm_global = torch.maximum(r2_perm_global, r2_perm_vec)
                else:
                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, int(perm_chunk)),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    betas, ses, tstats, r2_nominal_vec, r2_perm_vec = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=G_resid,
                        H_resid=H_resid,
                        k_eff=rez.k_eff if rez is not None else 1,
                        perm_stream=stream,
                        max_permutations=nperm,
                        return_nominal=True,
                        mixed_precision=mixed_precision,
                    )
                    p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                    dof = max(int(y_resid.shape[0]) - int(p_pred) - (rez.k_eff if rez is not None else 1), 1)
                    r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    t_g = tstats[:, 0].double()
                    if r2_max_t > best_r2_val:
                        best_r2_val = float(r2_max_t.item())
                        best_ix_var = ix
                        best_ix_pheno = j
                        best_beta = float(betas[ix, 0].item())
                        best_se = float(ses[ix, 0].item())
                        best_t = float(t_g[ix].item())
                        best_dof = int(dof)
                    r2_perm_global = torch.maximum(r2_perm_global, r2_perm_vec)

        if best_ix_var >= 0:
            if not dosage_mode:
                pval_perm = (
                    (r2_perm_global >= torch.tensor(best_r2_val, device=r2_perm_global.device)).sum().add_(1).float() / (r2_perm_global.numel() + 1)
                ).item()
                r2_perm_np = r2_perm_global.detach().cpu().numpy()
                if beta_approx:
                    pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                        r2_perm_np, best_r2_val, dof_init=best_dof
                    )
                else:
                    pval_beta = a_hat = b_hat = true_dof = p_true = np.nan
                pval_nominal = float(get_t_pval(best_t, best_dof))
        else:
            processed += 1
            continue

        pid_best = ids_list[best_ix_pheno]
        var_id = idx_to_id[v_idx[best_ix_var]]
        var_pos = int(pos_arr[v_idx[best_ix_var]])
        start_pos = ig.phenotype_start[pid_best]
        end_pos = ig.phenotype_end[pid_best]

        buffers = ensure_capacity(buffers, cursor, 1)
        row = {
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
            "true_dof": int(true_dof) if np.isfinite(true_dof) else int(best_dof),
            "pval_true_dof": p_true,
            "ma_samples": int(ma_samples_t[best_ix_var].item()),
            "ma_count": float(ma_count_t[best_ix_var].item()),
            "af": float(af_t[best_ix_var].item()),
        }
        if interaction_mode and interaction_columns:
            for anc_idx in range(len(interaction_columns) // 4):
                base = anc_idx * 4
                row[interaction_columns[base + 0]] = slope_gxh[anc_idx]
                row[interaction_columns[base + 1]] = slope_se_gxh[anc_idx]
                row[interaction_columns[base + 2]] = tstat_gxh[anc_idx]
                row[interaction_columns[base + 3]] = pval_gxh[anc_idx]
        if dosage_mode and dosage_vals is not None:
            for col in dosage_columns:
                row[col] = dosage_vals.get(col, np.nan)
        write_row(buffers, cursor, row)
        cursor += 1
        processed += 1
        if (
            logger.verbose
            and total_groups
            and progress_interval
            and (processed % progress_interval == 0 or processed == total_groups)
        ):
            logger.write(
                f"      processed {processed}/{total_groups} phenotype groups on {chrom_label}"
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
        perm_indices_mode: str = "generator", mixed_precision: str | None = None,
        ancestry_model: str = "main",
        interaction_covariate: Optional[object] = None,
        paired_covariate_df: Optional[pd.DataFrame] = None,
        min_ancestry_ac: float = 0.0,
) -> pd.DataFrame:
    """
    Empirical cis-QTL mapping (one top variant per phenotype) with permutations.
    Returns a DataFrame with empirical p-values (and optional Beta approximation).

    Parameters
    ----------
    preload_haplotypes : bool, default True
        When haplotypes are provided, pre-load them into a contiguous torch.Tensor
        on the requested device to avoid per-batch host<->device transfers.
    paired_covariate_df : DataFrame (phenotypes x samples), optional
        One phenotype-specific covariate added to the shared covariates for the
        phenotypes it lists (tensorQTL ``paired_covariate_df``); ungrouped only.
    ancestry_model : {"main", "interaction", "dosage"}
        ``"dosage"`` runs the ancestry-specific dosage scan: the per-variant
        statistic is the joint Wald F over the K ancestry effects and the
        permutation statistic is its maximum over the window, so ``pval_perm``
        and ``pval_beta`` calibrate the *joint* signal.  ``pval_nominal`` is
        then the joint p-value of the lead variant and the per-ancestry
        columns of :func:`localqtl.cis.map_nominal` are reported for it.
        With ``group_s`` the maximum is taken over the phenotypes of the
        group as well, which is valid because they share the residual
        degrees of freedom.
    min_ancestry_ac : float, default 0
        Dosage model: ancestries whose in-sample minor-allele count is below
        this value get NaN effect estimates and pairwise differences in the
        reported lead row (the joint test and the variant ranking are
        unchanged).
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
    if ancestry_model == "interaction":
        logger.write("  * ancestry model: interaction (GxH)")
    elif ancestry_model == "dosage":
        logger.write("  * ancestry model: ancestry-specific dosage "
                     "(permutations calibrate the joint F over ancestries)")
        if min_ancestry_ac and min_ancestry_ac > 0:
            logger.write(f"  * masking ancestry effects with in-sample minor-allele count < {min_ancestry_ac:g}")
    if interaction_covariate is not None:
        logger.write("  * covariate interaction: GxC")

    if nperm is None or nperm <= 0:
        raise ValueError("nperm must be a positive integer for map_permutations.")
    if paired_covariate_df is not None and group_s is not None:
        raise ValueError("paired_covariate_df is not supported with phenotype groups.")
    if paired_covariate_df is not None:
        logger.write("  * including phenotype-specific covariate")
    validate_ancestry_model(
        ancestry_model, haplotypes=haplotypes, interaction_covariate=interaction_covariate,
        allow_dosage=True, caller="map_permutations",
    )
    if ancestry_model == "dosage" and paired_covariate_df is not None:
        raise ValueError("ancestry_model='dosage' cannot be combined with paired_covariate_df.")
    if min_ancestry_ac and min_ancestry_ac > 0 and ancestry_model != "dosage":
        raise ValueError("min_ancestry_ac is only meaningful with ancestry_model='dosage'.")

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
        Y_resid, rez = residualize_matrix_with_covariates(Y, covariates_use,
                                                          device, tensorqtl_flavor)

    paired_ctx = _paired_covariate_context(paired_covariate_df, covariates_use, ig.phenotype_df.index,
                                           phenotype_df, device, tensorqtl_flavor)
    ig.phenotype_df = pd.DataFrame(Y_resid.cpu().numpy(), index=ig.phenotype_df.index,
                                   columns=ig.phenotype_df.columns)

    # Core either grouped or single-phenotype
    phenotype_counts = ig.phenotype_pos_df['chr'].value_counts().to_dict()
    total_phenotypes = int(ig.phenotype_df.shape[0])
    if logger.verbose:
        logger.write(f"    Mapping all chromosomes ({total_phenotypes} phenotypes)")

    group_mode = getattr(ig, "group_s", None) is not None
    core = _run_permutation_core_group if group_mode else _run_permutation_core
    n_anc = _n_ancestries(haplotypes, ancestry_model)

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
                        perm_chunk=perm_chunk,
                        perm_indices_mode=perm_indices_mode,
                        mixed_precision=mixed_precision,
                        ancestry_model=ancestry_model,
                        n_ancestries=n_anc,
                        interaction_covariate_t=interaction_cov_t,
                        min_ancestry_ac=min_ancestry_ac,
                    )
                else:
                    chrom_df = core(
                        ig, variant_df, rez, nperm=nperm, device=device,
                        beta_approx=beta_approx, maf_threshold=maf_threshold,
                        seed=seed, chrom=chrom,
                        logger=logger, total_phenotypes=total_units,
                        perm_chunk=perm_chunk,
                        perm_indices_mode=perm_indices_mode,
                        mixed_precision=mixed_precision,
                        ancestry_model=ancestry_model,
                        n_ancestries=n_anc,
                        interaction_covariate_t=interaction_cov_t,
                        paired_ctx=paired_ctx,
                        min_ancestry_ac=min_ancestry_ac,
                    )
            results.append(chrom_df)
            if logger.verbose:
                elapsed = time.time() - chrom_start
                logger.write(f"    Chromosome {chrom} completed in {elapsed / 60:.2f} min")

    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed permutation scan in {elapsed / 60:.2f} min")

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    return pd.DataFrame()
