import torch, time
import numpy as np
import pandas as pd
from typing import Optional, Sequence

try:
    torch.backends.fp32_precision = "ieee"
except Exception:
    pass

from ..utils import SimpleLogger, subseed
from .._torch_utils import to_device_tensor
from ..stats import beta_approx_pval, get_t_pval
from ..haplotypeio import InputGeneratorCis, InputGeneratorCisWithHaps
from ..preproc import impute_mean_and_filter, filter_by_maf
from ..regression_kernels import Residualizer, prep_ctx_for_perm, perm_chunk_r2
from ._permute import PermutationStream, compute_perm_r2_max
from ._buffer import (
    allocate_result_buffers,
    ensure_capacity, write_row,
    buffers_to_dataframe
)
from .common import (
    align_like_casefold,
    residualize_batch,
    residualize_batch_interaction,
    prepare_interaction_covariate,
    dosage_vector_for_covariate
)

__all__ = [
    "map_independent",
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

def _auto_perm_chunk(n_variants: int, nperm: int, pH: int = 0, safety: float = 0.8) -> int:
    if not torch.cuda.is_available() or n_variants <= 0:
        return min(nperm, 2048)
    free, _ = torch.cuda.mem_get_info()
    bytes_per_col = 4 * n_variants * (3 + 2 * max(pH, 0))
    if bytes_per_col <= 0:
        return min(nperm, 2048)
    max_chunk = int((free * safety) // bytes_per_col)
    if max_chunk <= 0:
        return min(nperm, 1024)
    p2 = 1 << (max_chunk.bit_length() - 1)
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
        chrom: str | None = None, perm_chunk: int = 2048, logger: SimpleLogger | None = None,
        total_items: int | None = None, item_label: str = "phenotypes",
        tensorqtl_flavor: bool = False, perm_indices_mode: str = "generator",
        mixed_precision: str | None = None, nperm_stage1: int | None = 2000,
        ancestry_model: str | None = None, n_ancestries: int | None = None,
        interaction_covariate_t: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Forward–backward independent mapping for ungrouped phenotypes."""
    expected_columns = [
        "phenotype_id", "variant_id", "start_distance", "end_distance",
        "num_var", "slope", "slope_se", "tstat", "r2_nominal", "pval_nominal",
        "pval_perm", "pval_beta", "beta_shape1", "beta_shape2", "ma_samples",
        "ma_count", "af", "true_dof", "pval_true_dof", "rank",
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
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
    if interaction_columns:
        for col in interaction_columns:
            dtype_map[col] = np.float32
    buffers = allocate_result_buffers(expected_columns, dtype_map, signif_seed_df.shape[0])
    cursor = 0
    processed = 0

    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, int(total_items) // 10) if total_items else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"

    perm_indices_mode = (perm_indices_mode or "generator").lower()
    if perm_indices_mode not in {"generator", "cpu_pinned", "gpu"}:
        perm_indices_mode = "generator"
    stage1_cap = None
    if nperm_stage1 is not None and nperm_stage1 > 0:
        stage1_cap = int(nperm_stage1)

    idx_to_id = variant_df.index.to_numpy()
    pos_arr = variant_df["pos"].to_numpy(np.int32)

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
            dtype=torch.float32,
        )

    var_in_frame = set(variant_df.index)

    def maybe_project_to_cov(rez_cov: Residualizer | None, vec: torch.Tensor) -> torch.Tensor:
        if rez_cov is None or rez_cov.rank == 0:
            return vec
        Q = rez_cov.Q_t
        coeff = torch.matmul(Q.T, vec)
        return vec - torch.matmul(Q, coeff)

    with torch.inference_mode():
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
                continue
            seed_row = signif_seed_df.loc[pid]

            y_t = to_device_tensor(p, device, dtype=torch.float32)
            G_t = to_device_tensor(G_block, device, dtype=torch.float32)
            H_t = None if H_block is None else to_device_tensor(H_block, device, dtype=torch.float32)

            G_imputed, keep_mask, _ = impute_mean_and_filter(G_t)
            if G_imputed.shape[0] == 0:
                continue

            mask_cpu = keep_mask.detach().cpu().numpy()
            v_idx = v_idx[mask_cpu]
            if H_t is not None:
                H_t = H_t[mask_cpu]
                if H_t.shape[2] > 1:
                    H_t = H_t[:, :, :-1]

            G_t = G_imputed

            if maf_threshold and maf_threshold > 0:
                keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
                if keep_maf.sum().item() == 0:
                    continue
                mask_cpu = keep_maf.detach().cpu().numpy()
                G_t = G_t[keep_maf]
                v_idx = v_idx[mask_cpu]
                if H_t is not None:
                    H_t = H_t[keep_maf]

            interaction_mode = ancestry_model == "interaction" and H_t is not None
            covar_interaction = interaction_covariate_t is not None
            covar_interaction = interaction_covariate_t is not None

            pH = 0 if H_t is None else int(H_t.shape[2])
            perm_chunk_local = _auto_perm_chunk(G_t.shape[0], nperm, pH=pH)
            if perm_chunk is not None and perm_chunk > 0:
                perm_chunk_local = min(perm_chunk_local, int(perm_chunk))

            rez_cov = Residualizer(covariates_base_t, tensorqtl_flavor=tensorqtl_flavor) if covariates_base_t is not None else None
            y0, G0, H0 = residualize_batch(y_t, G_t, H_t, rez_cov, center=True, group=False)
            k_base = rez_cov.k_eff if rez_cov is not None else 0

            perm_seed_base = subseed(seed, pid) if seed is not None else None
            gen = None
            if random_tiebreak and seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(perm_seed_base)

            dosage_proj: dict[str, torch.Tensor] = {}

            def ensure_projected(var_id: str) -> torch.Tensor:
                if var_id in dosage_proj:
                    return dosage_proj[var_id]
                if var_id not in ig.genotype_df.index:
                    raise KeyError(f"Variant {var_id} not in genotype_df for covariate projection")
                raw = dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=var_id,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                )
                vec = torch.from_numpy(raw)
                if device == "cuda":
                    vec = vec.pin_memory()
                vec_t = vec.to(y0.device, non_blocking=True).to(torch.float32)
                proj = maybe_project_to_cov(rez_cov, vec_t)
                dosage_proj[var_id] = proj
                return proj

            seed_vid = seed_row.get("variant_id")
            if isinstance(seed_vid, str) and seed_vid in var_in_frame and seed_vid in ig.genotype_df.index:
                ensure_projected(seed_vid)

            def evaluate_model(extra_ids: Sequence[str]):
                extras = [ensure_projected(v) for v in extra_ids]
                rez_extra = None
                if extras:
                    extras_mat = torch.stack(extras, dim=1)
                    rez_extra = Residualizer(extras_mat, tensorqtl_flavor=tensorqtl_flavor)
                    if interaction_mode:
                        y_resid, G_resid, H_resid, I_resid = residualize_batch_interaction(
                            y0, G0, H0, rez_extra, center=True, group=False,
                        )
                    else:
                        y_resid, G_resid, H_resid = residualize_batch(
                            y0, G0, H0, rez_extra, center=True, group=False,
                        )
                else:
                    if interaction_mode:
                        y_resid, G_resid, H_resid, I_resid = residualize_batch_interaction(
                            y0, G0, H0, None, center=True, group=False,
                        )
                    else:
                        y_resid, G_resid, H_resid = y0, G0, H0
                k_eff = k_base + (rez_extra.k_eff if rez_extra is not None else 0)
                cap = nperm if stage1_cap is None else min(nperm, stage1_cap)
                if interaction_mode:
                    if H_resid is None:
                        raise ValueError("interaction model requires haplotypes.")
                    pH = int(H_resid.shape[2])
                    n_anc = int(n_ancestries) if n_ancestries is not None else pH
                    H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid], dim=2)
                    n_samples_local = int(y_resid.shape[0])
                    p_pred = 1 + int(H_cov.shape[2])
                    dof = max(n_samples_local - int(k_eff) - int(p_pred), 1)

                    def _perm_chunks(stream: PermutationStream, count: int) -> list[torch.Tensor]:
                        return list(stream.iter_chunks(count))

                    def _r2_perm_from_chunks(
                        y_vec: torch.Tensor,
                        chunks: list[torch.Tensor],
                        ctx_list: list[dict],
                        g_list: list[torch.Tensor],
                    ) -> torch.Tensor:
                        if not chunks:
                            return torch.empty((0,), device=y_vec.device, dtype=torch.float32)
                        blocks = []
                        n_samp = int(y_vec.shape[0])
                        for sel in chunks:
                            flat = sel.reshape(-1)
                            y_perm = y_vec.index_select(0, flat).view(sel.shape[0], n_samp).transpose(0, 1)
                            r2_max = None
                            for ctx_k, g_k in zip(ctx_list, g_list):
                                r2_block = perm_chunk_r2(ctx_k, H_cov, g_k, y_perm, mixed_precision=mixed_precision)
                                r2_max = r2_block if r2_max is None else torch.maximum(r2_max, r2_block)
                            blocks.append(r2_max.to(torch.float32))
                        return torch.cat(blocks, dim=0)

                    anc_results = []
                    r2_perm = None
                    best_r2 = -float("inf")
                    best_ix = -1
                    best_anc = -1

                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, perm_chunk_local),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    stage_chunks = _perm_chunks(stream, cap)

                    ctx_list: list[dict] = []
                    g_list: list[torch.Tensor] = []

                    for k in range(pH):
                        ctx, betas, ses, tstats = prep_ctx_for_perm(
                            y_resid, I_resid[:, :, k], H_cov, k_eff, mixed_precision=mixed_precision,
                        )
                        tvals = tstats[:, 0].double()
                        t2 = tvals.pow(2)
                        dof_ctx = ctx.get("dof", dof)
                        r2_nominal_vec = (t2 / (t2 + float(dof_ctx))).to(torch.float32)
                        r2_nominal_vec = torch.nan_to_num(r2_nominal_vec, nan=-1.0)
                        r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                        ix = int(ix_t.item())
                        if random_tiebreak:
                            ties = torch.nonzero(torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12), as_tuple=True)[0]
                            if ties.numel() > 1:
                                if gen is None:
                                    choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                                else:
                                    choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device, generator=gen)
                                ix = int(ties[int(choice_tensor.item())].item())
                                r2_max_t = r2_nominal_vec[ix]

                        anc_results.append((betas, ses, tstats))
                        if r2_max_t > best_r2:
                            best_r2 = float(r2_max_t.item())
                            best_ix = ix
                            best_anc = k
                        ctx_list.append(ctx)
                        g_list.append(I_resid[:, :, k])

                    r2_perm = _r2_perm_from_chunks(y_resid, stage_chunks, ctx_list, g_list)

                    slope_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                    slope_se_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                    tstat_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                    pval_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                    for k in range(pH):
                        betas, ses, tstats = anc_results[k]
                        slope_gxh[k] = float(betas[best_ix, 0].item())
                        slope_se_gxh[k] = float(ses[best_ix, 0].item())
                        tstat_gxh[k] = float(tstats[best_ix, 0].item())
                        pval_gxh[k] = float(get_t_pval(tstat_gxh[k], dof))

                    beta = float(slope_gxh[best_anc])
                    se = float(slope_se_gxh[best_anc])
                    tval = float(tstat_gxh[best_anc])
                    r2_nom = float(best_r2)
                    pval_nominal = float(get_t_pval(tval, dof))

                    pval_perm = (
                        (r2_perm >= torch.tensor(best_r2, device=r2_perm.device)).sum().add_(1).float() / (r2_perm.numel() + 1)
                    ).item()

                    if beta_approx:
                        r2_perm_np = r2_perm.detach().cpu().numpy()
                        pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                            r2_perm_np, r2_nom, dof_init=dof,
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof = p_true = np.nan

                    stop_pval = float(pval_beta)
                    if not np.isfinite(stop_pval):
                        stop_pval = pval_perm
                elif covar_interaction:
                    c_t = interaction_covariate_t
                    if c_t.device != G_resid.device:
                        c_t = c_t.to(G_resid.device)
                    I_t = G_resid * c_t
                    c_rep = c_t.unsqueeze(0).expand(G_resid.shape[0], -1).unsqueeze(-1)
                    if H_resid is None:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), c_rep], dim=2)
                    else:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid, c_rep], dim=2)

                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, perm_chunk_local),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    betas, ses, tstats, r2_nominal_vec, r2_perm_vals = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=I_t,
                        H_resid=H_cov,
                        k_eff=k_eff,
                        perm_stream=stream,
                        max_permutations=cap,
                        return_nominal=True,
                        mixed_precision=mixed_precision,
                    )
                    if r2_nominal_vec is None:
                        r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid.device, dtype=torch.float32)
                    r2_nominal_vec = torch.nan_to_num(r2_nominal_vec.to(torch.float32), nan=-1.0)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    if random_tiebreak:
                        ties = torch.nonzero(torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12), as_tuple=True)[0]
                        if ties.numel() > 1:
                            if gen is None:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                            else:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device, generator=gen)
                            ix = int(ties[int(choice_tensor.item())].item())
                            r2_max_t = r2_nominal_vec[ix]

                    n_samples_local = int(y_resid.shape[0])
                    p_pred = 1 + int(H_cov.shape[2])
                    dof = max(n_samples_local - int(k_eff) - int(p_pred), 1)

                    beta = float(betas[ix, 0].item())
                    se = float(ses[ix, 0].item())
                    tval = float(tstats[ix, 0].item())
                    r2_nom = float(r2_max_t.item())
                    pval_nominal = float(get_t_pval(tval, dof))

                    r2_perm = r2_perm_vals
                    pval_perm = (
                        (r2_perm >= r2_max_t).sum().add_(1).float() / (r2_perm.numel() + 1)
                    ).item()

                    if beta_approx:
                        r2_perm_np = r2_perm.detach().cpu().numpy()
                        pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                            r2_perm_np, r2_nom, dof_init=dof,
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof = p_true = np.nan

                    stop_pval = float(pval_beta)
                    if not np.isfinite(stop_pval):
                        stop_pval = pval_perm
                else:
                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, perm_chunk_local),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    betas, ses, tstats, r2_nominal_vec, r2_perm_vals = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=G_resid,
                        H_resid=H_resid,
                        k_eff=k_eff,
                        perm_stream=stream,
                        max_permutations=cap,
                        return_nominal=True,
                        mixed_precision=mixed_precision,
                    )
                    if r2_nominal_vec is None:
                        r2_nominal_vec = torch.zeros(G_resid.shape[0], device=y_resid.device, dtype=torch.float32)
                    r2_nominal_vec = torch.nan_to_num(r2_nominal_vec.to(torch.float32), nan=-1.0)
                    r2_max_t, ix_t = _nanmax(r2_nominal_vec, dim=0)
                    ix = int(ix_t.item())
                    if random_tiebreak:
                        ties = torch.nonzero(torch.isclose(r2_nominal_vec, r2_max_t, atol=1e-12), as_tuple=True)[0]
                        if ties.numel() > 1:
                            if gen is None:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                            else:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device, generator=gen)
                            ix = int(ties[int(choice_tensor.item())].item())
                            r2_max_t = r2_nominal_vec[ix]

                    n_samples_local = int(y_resid.shape[0])
                    p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)
                    dof = max(n_samples_local - int(k_eff) - int(p_pred), 1)

                    beta = float(betas[ix, 0].item())
                    se = float(ses[ix, 0].item())
                    tval = float(tstats[ix, 0].item())
                    r2_nom = float(r2_max_t.item())
                    pval_nominal = float(get_t_pval(tval, dof))

                    r2_perm = r2_perm_vals
                    pval_perm = (
                        (r2_perm >= r2_max_t).sum().add_(1).float() / (r2_perm.numel() + 1)
                    ).item()

                    if beta_approx:
                        r2_perm_np = r2_perm.detach().cpu().numpy()
                        pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                            r2_perm_np, r2_nom, dof_init=dof,
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof = p_true = np.nan

                    stop_pval = float(pval_beta)
                    if not np.isfinite(stop_pval):
                        stop_pval = pval_perm

                if r2_perm.numel() < nperm and stop_pval <= signif_threshold * 2:
                    remaining = nperm - r2_perm.numel()
                    if remaining > 0:
                        if interaction_mode:
                            extra_chunks = _perm_chunks(stream, remaining)
                            extra_perm = _r2_perm_from_chunks(y_resid, extra_chunks, ctx_list, g_list)
                        else:
                            _, _, _, _, extra_perm = compute_perm_r2_max(
                                y_resid=y_resid,
                                G_resid=G_resid,
                                H_resid=H_resid,
                                k_eff=k_eff,
                                perm_stream=stream,
                                max_permutations=remaining,
                                return_nominal=False,
                                mixed_precision=mixed_precision,
                            )
                        if extra_perm.numel():
                            r2_perm = torch.cat([r2_perm, extra_perm])
                            pval_perm = (
                                (r2_perm >= (best_r2 if interaction_mode else r2_max_t)).sum().add_(1).float() / (r2_perm.numel() + 1)
                            ).item()
                            if beta_approx:
                                r2_perm_np = r2_perm.detach().cpu().numpy()
                                pval_beta, a_hat, b_hat, true_dof, p_true = beta_approx_pval(
                                    r2_perm_np, r2_nom, dof_init=dof,
                                )
                            else:
                                pval_beta = a_hat = b_hat = true_dof = p_true = np.nan
                            stop_pval = float(pval_beta)
                            if not np.isfinite(stop_pval):
                                stop_pval = pval_perm

                return {
                    "betas": None if interaction_mode else betas,
                    "ses": None if interaction_mode else ses,
                    "tstats": None if interaction_mode else tstats,
                    "r2_nominal_vec": None if interaction_mode else r2_nominal_vec,
                    "r2_perm": r2_perm,
                    "ix": best_ix if interaction_mode else ix,
                    "beta": beta,
                    "se": se,
                    "tval": tval,
                    "r2_nom": r2_nom,
                    "pval_nominal": pval_nominal,
                    "pval_perm": pval_perm,
                    "pval_beta": float(pval_beta),
                    "a_hat": float(a_hat),
                    "b_hat": float(b_hat),
                    "true_dof": int(true_dof) if np.isfinite(true_dof) else int(dof),
                    "p_true": float(p_true),
                    "stop_pval": float(stop_pval),
                    "num_var": int(G_resid.shape[0]),
                    "dof": int(dof),
                    "slope_gxh": slope_gxh if interaction_mode else None,
                    "slope_se_gxh": slope_se_gxh if interaction_mode else None,
                    "tstat_gxh": tstat_gxh if interaction_mode else None,
                    "pval_gxh": pval_gxh if interaction_mode else None,
                    "perm_stream": stream if interaction_mode else None,
                    "perm_ctx_list": ctx_list if interaction_mode else None,
                    "perm_g_list": g_list if interaction_mode else None,
                    "perm_chunks": stage_chunks if interaction_mode else None,
                }

            forward_records: list[dict[str, object]] = []
            base_record = {col: seed_row.get(col) for col in expected_columns if col != "rank"}
            forward_records.append(base_record)

            while True:
                extras_ids = list(dosage_proj.keys())
                eval_res = evaluate_model(extras_ids)
                if eval_res["stop_pval"] > signif_threshold:
                    break

                ix = eval_res["ix"]
                var_id = idx_to_id[v_idx[ix]]
                var_pos = int(pos_arr[v_idx[ix]])
                start_pos = ig.phenotype_start[pid]
                end_pos = ig.phenotype_end[pid]

                g_sel = G_t[ix]
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

                forward_records.append({
                    "phenotype_id": pid,
                    "variant_id": var_id,
                    "start_distance": int(var_pos - start_pos),
                    "end_distance": int(var_pos - end_pos),
                    "num_var": eval_res["num_var"],
                    "slope": eval_res["beta"],
                    "slope_se": eval_res["se"],
                    "tstat": eval_res["tval"],
                    "r2_nominal": eval_res["r2_nom"],
                    "pval_nominal": eval_res["pval_nominal"],
                    "pval_perm": eval_res["pval_perm"],
                    "pval_beta": eval_res["pval_beta"],
                    "beta_shape1": eval_res["a_hat"],
                    "beta_shape2": eval_res["b_hat"],
                    "af": af,
                    "ma_samples": ma_samples,
                    "ma_count": ma_count,
                    "true_dof": eval_res["true_dof"],
                    "pval_true_dof": eval_res["p_true"],
                })
                if interaction_mode and interaction_columns:
                    for anc_idx in range(len(interaction_columns) // 4):
                        base = anc_idx * 4
                        forward_records[-1][interaction_columns[base + 0]] = eval_res["slope_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 1]] = eval_res["slope_se_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 2]] = eval_res["tstat_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 3]] = eval_res["pval_gxh"][anc_idx]

                if var_id in var_in_frame and var_id in ig.genotype_df.index and var_id not in dosage_proj:
                    ensure_projected(var_id)

            if not forward_records:
                continue

            if len(forward_records) > 1:
                kept_records: list[dict[str, object]] = []
                selected = [rec["variant_id"] for rec in forward_records]
                for rk, drop_vid in enumerate(selected, start=1):
                    kept = [v for v in selected if v != drop_vid]
                    eval_res = evaluate_model(kept)
                    if eval_res["stop_pval"] <= signif_threshold:
                        pid_best = pid
                        best_idx = eval_res["ix"]
                        var_id = idx_to_id[v_idx[best_idx]]
                        var_pos = int(pos_arr[v_idx[best_idx]])
                        start_pos = ig.phenotype_start[pid_best]
                        end_pos = ig.phenotype_end[pid_best]

                        g_sel = G_t[best_idx]
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

                        kept_records.append({
                            "phenotype_id": pid_best,
                            "variant_id": var_id,
                            "start_distance": int(var_pos - start_pos),
                            "end_distance": int(var_pos - end_pos),
                            "num_var": eval_res["num_var"],
                            "slope": eval_res["beta"],
                            "slope_se": eval_res["se"],
                            "tstat": eval_res["tval"],
                            "r2_nominal": eval_res["r2_nom"],
                            "pval_nominal": eval_res["pval_nominal"],
                            "pval_perm": eval_res["pval_perm"],
                            "pval_beta": eval_res["pval_beta"],
                            "beta_shape1": eval_res["a_hat"],
                            "beta_shape2": eval_res["b_hat"],
                            "af": af,
                            "ma_samples": ma_samples,
                            "ma_count": ma_count,
                            "true_dof": eval_res["true_dof"],
                            "pval_true_dof": eval_res["p_true"],
                            "rank": int(rk),
                        })
                        if interaction_mode and interaction_columns:
                            for anc_idx in range(len(interaction_columns) // 4):
                                base = anc_idx * 4
                                kept_records[-1][interaction_columns[base + 0]] = eval_res["slope_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 1]] = eval_res["slope_se_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 2]] = eval_res["tstat_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 3]] = eval_res["pval_gxh"][anc_idx]

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
        chrom: str | None = None, perm_chunk: int = 2048, logger: SimpleLogger | None = None,
        total_items: int | None = None, item_label: str = "phenotype groups",
        tensorqtl_flavor: bool = False, perm_indices_mode: str = "generator",
        mixed_precision: str | None = None, nperm_stage1: int | None = 2000,
        ancestry_model: str | None = None, n_ancestries: int | None = None,
        interaction_covariate_t: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Forward-backward independent mapping for grouped phenotypes."""
    expected_columns = [
        "group_id", "group_size", "phenotype_id", "variant_id", "start_distance",
        "end_distance", "num_var", "slope", "slope_se", "tstat", "r2_nominal",
        "pval_nominal", "pval_perm", "pval_beta", "beta_shape1", "beta_shape2",
        "ma_samples", "ma_count", "af", "true_dof", "pval_true_dof", "rank",
    ]
    interaction_columns: list[str] = []
    if ancestry_model == "interaction" and n_ancestries is not None and n_ancestries > 0:
        interaction_columns = _interaction_columns(int(n_ancestries))
        expected_columns.extend(interaction_columns)
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
    if interaction_columns:
        for col in interaction_columns:
            dtype_map[col] = np.float32
    buffers = allocate_result_buffers(expected_columns, dtype_map, seed_by_group_df.shape[0])
    cursor = 0
    processed = 0

    if logger is None:
        logger = SimpleLogger(verbose=True, timestamps=True)
    progress_interval = max(1, int(total_items) // 10) if total_items else 0
    chrom_label = f"{chrom}" if chrom is not None else "all chromosomes"

    perm_indices_mode = (perm_indices_mode or "generator").lower()
    if perm_indices_mode not in {"generator", "cpu_pinned", "gpu"}:
        perm_indices_mode = "generator"
    stage1_cap = None
    if nperm_stage1 is not None and nperm_stage1 > 0:
        stage1_cap = int(nperm_stage1)

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
            dtype=torch.float32,
        )

    def maybe_project_to_cov(rez_cov: Residualizer | None, vec: torch.Tensor) -> torch.Tensor:
        if rez_cov is None or rez_cov.rank == 0:
            return vec
        Q = rez_cov.Q_t
        coeff = torch.matmul(Q.T, vec)
        return vec - torch.matmul(Q, coeff)

    with torch.inference_mode():
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
            H_t = None if H_block is None else to_device_tensor(H_block, device, dtype=torch.float32)

            G_imputed, keep_mask, _ = impute_mean_and_filter(G_t)
            if G_imputed.shape[0] == 0:
                continue

            mask_cpu = keep_mask.detach().cpu().numpy()
            v_idx = v_idx[mask_cpu]
            if H_t is not None:
                H_t = H_t[mask_cpu]
                if H_t.shape[2] > 1:
                    H_t = H_t[:, :, :-1]

            if not G_imputed.is_contiguous():
                G_t = G_imputed.contiguous()
            else:
                G_t = G_imputed

            if maf_threshold and maf_threshold > 0:
                keep_maf, _ = filter_by_maf(G_t, maf_threshold, ploidy=2)
                if keep_maf.sum().item() == 0:
                    continue
                mask_cpu = keep_maf.detach().cpu().numpy()
                G_t = G_t[keep_maf]
                v_idx = v_idx[mask_cpu]
                if H_t is not None:
                    H_t = H_t[keep_maf]

            interaction_mode = ancestry_model == "interaction" and H_t is not None

            pH = 0 if H_t is None else int(H_t.shape[2])
            perm_chunk_local = _auto_perm_chunk(G_t.shape[0], nperm, pH=pH)
            if perm_chunk is not None and perm_chunk > 0:
                perm_chunk_local = min(perm_chunk_local, int(perm_chunk))

            ids_list = list(ids)
            y_stack = torch.stack([
                to_device_tensor(
                    ig.phenotype_df.loc[pid].to_numpy(np.float32, copy=False), device,
                    dtype=torch.float32,
                )
                for pid in ids_list
            ], dim=0)

            rez_cov = Residualizer(covariates_base_t, tensorqtl_flavor=tensorqtl_flavor) if covariates_base_t is not None else None
            y_base_list, G0, H0 = residualize_batch(y_stack, G_t, H_t, rez_cov, center=True, group=True)
            k_base = rez_cov.k_eff if rez_cov is not None else 0

            perm_seed_base = subseed(seed, f"group:{group_id}") if seed is not None else None
            gen = None
            if random_tiebreak and seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(perm_seed_base)

            dosage_proj: dict[str, torch.Tensor] = {}

            def ensure_projected(var_id: str) -> torch.Tensor:
                if var_id in dosage_proj:
                    return dosage_proj[var_id]
                if var_id not in geno_has_variant:
                    raise KeyError(f"Variant {var_id} not found in genotype_df for covariate projection")
                raw = dosage_vector_for_covariate(
                    genotype_df=ig.genotype_df,
                    variant_id=var_id,
                    sample_order=ig.phenotype_df.columns,
                    missing=missing,
                )
                vec = torch.from_numpy(raw)
                if device == "cuda":
                    vec = vec.pin_memory()
                vec_t = vec.to(G0.device, non_blocking=True).to(torch.float32)
                proj = maybe_project_to_cov(rez_cov, vec_t)
                dosage_proj[var_id] = proj
                return proj

            if seed_vid in var_in_frame and seed_vid in geno_has_variant:
                ensure_projected(seed_vid)

            def evaluate_group_model(extra_ids: Sequence[str]) -> dict[str, object]:
                extras = [ensure_projected(v) for v in extra_ids]
                rez_extra = None
                if extras:
                    extras_mat = torch.stack(extras, dim=1)
                    rez_extra = Residualizer(extras_mat, tensorqtl_flavor=tensorqtl_flavor)
                    if interaction_mode:
                        y_resid_list, G_resid, H_resid, I_resid = residualize_batch_interaction(
                            y_base_list, G0, H0, rez_extra, center=True, group=True,
                        )
                    else:
                        y_resid_list, G_resid, H_resid = residualize_batch(
                            y_base_list, G0, H0, rez_extra, center=True, group=True,
                        )
                else:
                    if interaction_mode:
                        y_resid_list, G_resid, H_resid, I_resid = residualize_batch_interaction(
                            y_base_list, G0, H0, None, center=True, group=True,
                        )
                    else:
                        y_resid_list, G_resid, H_resid = y_base_list, G0, H0
                k_eff = k_base + (rez_extra.k_eff if rez_extra is not None else 0)
                num_var = int(G_resid.shape[0])
                stage_cap = nperm if stage1_cap is None else min(nperm, stage1_cap)

                if interaction_mode:
                    if H_resid is None:
                        raise ValueError("interaction model requires haplotypes.")
                    pH = int(H_resid.shape[2])
                    n_anc = int(n_ancestries) if n_ancestries is not None else pH
                    H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid], dim=2)
                    p_pred = 1 + int(H_cov.shape[2])

                    def _perm_chunks(stream: PermutationStream, count: int) -> list[torch.Tensor]:
                        return list(stream.iter_chunks(count))

                    def _r2_perm_from_chunks(
                        y_vec: torch.Tensor,
                        chunks: list[torch.Tensor],
                        ctx_list: list[dict],
                        g_list: list[torch.Tensor],
                    ) -> torch.Tensor:
                        if not chunks:
                            return torch.empty((0,), device=y_vec.device, dtype=torch.float32)
                        blocks = []
                        n_samp = int(y_vec.shape[0])
                        for sel in chunks:
                            flat = sel.reshape(-1)
                            y_perm = y_vec.index_select(0, flat).view(sel.shape[0], n_samp).transpose(0, 1)
                            r2_max = None
                            for ctx_k, g_k in zip(ctx_list, g_list):
                                r2_block = perm_chunk_r2(ctx_k, H_cov, g_k, y_perm, mixed_precision=mixed_precision)
                                r2_max = r2_block if r2_max is None else torch.maximum(r2_max, r2_block)
                            blocks.append(r2_max.to(torch.float32))
                        return torch.cat(blocks, dim=0)

                    per_pheno: list[dict[str, object]] = []
                    r2_perm_tensors: list[torch.Tensor] = []
                    best_tensor = torch.tensor(float("-inf"), device=G_resid.device, dtype=torch.float32)
                    best_idx = -1

                    for j, (pid_inner, y_resid) in enumerate(zip(ids_list, y_resid_list)):
                        best_r2_pheno = -float("inf")
                        best_ix_var_pheno = -1
                        best_anc_pheno = -1
                        anc_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                        stream = PermutationStream(
                            n_samples=int(y_resid.shape[0]),
                            nperm=nperm,
                            device=y_resid.device,
                            chunk_size=max(1, perm_chunk_local),
                            seed=perm_seed_base,
                            mode=perm_indices_mode,
                        )
                        stage_chunks = _perm_chunks(stream, stage_cap)

                        ctx_list: list[dict] = []
                        g_list: list[torch.Tensor] = []

                        for k in range(pH):
                            ctx, betas, ses, tstats = prep_ctx_for_perm(
                                y_resid, I_resid[:, :, k], H_cov, k_eff, mixed_precision=mixed_precision,
                            )
                            tvals = tstats[:, 0].double()
                            t2 = tvals.pow(2)
                            dof_ctx = ctx.get("dof", max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1))
                            r2_nom_vec = (t2 / (t2 + float(dof_ctx))).to(torch.float32)
                            r2_nom_vec = torch.nan_to_num(r2_nom_vec.to(torch.float32), nan=-1.0)
                            r2_max_t, ix_t = _nanmax(r2_nom_vec, dim=0)
                            ix = int(ix_t.item())
                            anc_results.append((betas, ses, tstats))

                            if r2_max_t > best_r2_pheno:
                                best_r2_pheno = float(r2_max_t.item())
                                best_ix_var_pheno = ix
                                best_anc_pheno = k

                            ctx_list.append(ctx)
                            g_list.append(I_resid[:, :, k])

                        r2_perm_pheno = _r2_perm_from_chunks(y_resid, stage_chunks, ctx_list, g_list)

                        dof = max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1)
                        slope_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        slope_se_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        tstat_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        pval_gxh = np.full((n_anc,), np.nan, dtype=np.float32)
                        for k in range(pH):
                            betas, ses, tstats = anc_results[k]
                            slope_gxh[k] = float(betas[best_ix_var_pheno, 0].item())
                            slope_se_gxh[k] = float(ses[best_ix_var_pheno, 0].item())
                            tstat_gxh[k] = float(tstats[best_ix_var_pheno, 0].item())
                            pval_gxh[k] = float(get_t_pval(tstat_gxh[k], dof))

                        per_pheno.append({
                            "pid": pid_inner,
                            "ix": best_ix_var_pheno,
                            "anc": best_anc_pheno,
                            "tensor": best_r2_pheno,
                            "r2_perm": r2_perm_pheno,
                            "slope_gxh": slope_gxh,
                            "slope_se_gxh": slope_se_gxh,
                            "tstat_gxh": tstat_gxh,
                            "pval_gxh": pval_gxh,
                            "dof": dof,
                            "stream": stream,
                            "ctx_list": ctx_list,
                            "g_list": g_list,
                            "y_resid": y_resid,
                        })
                        r2_perm_tensors.append(r2_perm_pheno)

                        if torch.tensor(best_r2_pheno, device=best_tensor.device) > best_tensor:
                            best_tensor = torch.tensor(best_r2_pheno, device=best_tensor.device)
                            best_idx = j

                    r2_perm_max = torch.stack(r2_perm_tensors, dim=0).amax(dim=0)
                    best_info = per_pheno[best_idx] if best_idx >= 0 else None
                    pval_perm = float("inf")
                    pval_beta = float("nan")
                    a_hat = float("nan")
                    b_hat = float("nan")
                    p_true = float("nan")
                    true_dof_val = float("nan")
                    stop_pval = float("inf")

                    if best_info is not None:
                        pval_perm = (
                            (r2_perm_max >= torch.tensor(best_info["tensor"], device=r2_perm_max.device)).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                        ).item()
                        if beta_approx:
                            r2_perm_np = r2_perm_max.detach().cpu().numpy()
                            pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                                r2_perm_np, float(best_info["tensor"]), dof_init=best_info["dof"],
                            )
                        else:
                            pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                        stop_pval = float(pval_beta)
                        if not np.isfinite(stop_pval):
                            stop_pval = pval_perm

                    if best_info is not None and stage_cap < nperm and stop_pval <= signif_threshold * 2:
                        remaining = nperm - r2_perm_max.numel()
                        if remaining > 0:
                            updated_vectors: list[torch.Tensor] = []
                            for info in per_pheno:
                                extra_chunks = _perm_chunks(info["stream"], remaining)
                                extra_perm = _r2_perm_from_chunks(
                                    info["y_resid"], extra_chunks, info["ctx_list"], info["g_list"]
                                )
                                if extra_perm.numel():
                                    info["r2_perm"] = torch.cat([info["r2_perm"], extra_perm])
                                updated_vectors.append(info["r2_perm"])
                            if updated_vectors:
                                r2_perm_max = torch.stack(updated_vectors, dim=0).amax(dim=0)
                                pval_perm = (
                                    (r2_perm_max >= torch.tensor(best_info["tensor"], device=r2_perm_max.device)).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                                ).item()
                                if beta_approx:
                                    r2_perm_np = r2_perm_max.detach().cpu().numpy()
                                    pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                                        r2_perm_np, float(best_info["tensor"]), dof_init=best_info["dof"],
                                    )
                                else:
                                    pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                                stop_pval = float(pval_beta)
                                if not np.isfinite(stop_pval):
                                    stop_pval = pval_perm

                    if best_info is None:
                        return {
                            "best_ix_var": -1,
                            "best_pheno_idx": -1,
                            "stop_pval": float("inf"),
                        }

                    best_pid = per_pheno[best_idx]["pid"]
                    best_ix = per_pheno[best_idx]["ix"]
                    best_anc = per_pheno[best_idx]["anc"]
                    best_beta = float(per_pheno[best_idx]["slope_gxh"][best_anc])
                    best_se = float(per_pheno[best_idx]["slope_se_gxh"][best_anc])
                    best_t = float(per_pheno[best_idx]["tstat_gxh"][best_anc])
                    best_dof = per_pheno[best_idx]["dof"]
                    best_r2 = float(per_pheno[best_idx]["tensor"])
                    pval_nominal = float(get_t_pval(best_t, best_dof))

                    true_dof_out = int(true_dof_val) if np.isfinite(true_dof_val) else int(best_dof)

                    return {
                        "best_ix_var": int(best_ix),
                        "best_pheno_idx": int(best_idx),
                        "best_pid": best_pid,
                        "beta": best_beta,
                        "se": best_se,
                        "tval": best_t,
                        "r2_nom": best_r2,
                        "pval_nominal": pval_nominal,
                        "pval_perm": float(pval_perm),
                        "pval_beta": float(pval_beta),
                        "a_hat": float(a_hat),
                        "b_hat": float(b_hat),
                        "p_true": float(p_true),
                        "true_dof": true_dof_out,
                        "stop_pval": float(stop_pval),
                        "num_var": num_var,
                        "dof": best_dof,
                        "slope_gxh": per_pheno[best_idx]["slope_gxh"],
                        "slope_se_gxh": per_pheno[best_idx]["slope_se_gxh"],
                        "tstat_gxh": per_pheno[best_idx]["tstat_gxh"],
                        "pval_gxh": per_pheno[best_idx]["pval_gxh"],
                    }
                if covar_interaction:
                    c_t = interaction_covariate_t
                    if c_t.device != G_resid.device:
                        c_t = c_t.to(G_resid.device)
                    I_t = G_resid * c_t
                    c_rep = c_t.unsqueeze(0).expand(G_resid.shape[0], -1).unsqueeze(-1)
                    if H_resid is None:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), c_rep], dim=2)
                    else:
                        H_cov = torch.cat([G_resid.unsqueeze(-1), H_resid, c_rep], dim=2)

                    per_pheno: list[dict[str, object]] = []
                    r2_perm_tensors: list[torch.Tensor] = []
                    best_tensor = torch.tensor(float("-inf"), device=G_resid.device, dtype=torch.float32)
                    best_idx = -1

                    p_pred = 1 + int(H_cov.shape[2])

                    for j, (pid_inner, y_resid) in enumerate(zip(ids_list, y_resid_list)):
                        stream = PermutationStream(
                            n_samples=int(y_resid.shape[0]),
                            nperm=nperm,
                            device=y_resid.device,
                            chunk_size=max(1, perm_chunk_local),
                            seed=perm_seed_base,
                            mode=perm_indices_mode,
                        )
                        betas, ses, tstats, r2_nom_vec, r2_perm_stage1 = compute_perm_r2_max(
                            y_resid=y_resid,
                            G_resid=I_t,
                            H_resid=H_cov,
                            k_eff=k_eff,
                            perm_stream=stream,
                            max_permutations=stage_cap,
                            return_nominal=True,
                            mixed_precision=mixed_precision,
                        )
                        if r2_nom_vec is None:
                            r2_nom_vec = torch.zeros(num_var, device=y_resid.device, dtype=torch.float32)
                        r2_nom_vec = torch.nan_to_num(r2_nom_vec.to(torch.float32), nan=-1.0)
                        r2_max_t, ix_t = _nanmax(r2_nom_vec, dim=0)
                        ix = int(ix_t.item())
                        if random_tiebreak:
                            ties = torch.nonzero(torch.isclose(r2_nom_vec, r2_max_t, atol=1e-12), as_tuple=True)[0]
                            if ties.numel() > 1:
                                if gen is None:
                                    choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                                else:
                                    choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device, generator=gen)
                                ix = int(ties[int(choice_tensor.item())].item())
                                r2_max_t = r2_nom_vec[ix]

                        dof = max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1)
                        beta_val = float(betas[ix, 0].item())
                        se_val = float(ses[ix, 0].item())
                        tval = float(tstats[ix, 0].item())

                        per_pheno.append({
                            "pid": pid_inner,
                            "stream": stream,
                            "betas": betas,
                            "ses": ses,
                            "tstats": tstats,
                            "r2_nominal_vec": r2_nom_vec,
                            "ix": ix,
                            "tensor": r2_max_t,
                            "beta": beta_val,
                            "se": se_val,
                            "tval": tval,
                            "dof": dof,
                            "r2_perm": r2_perm_stage1,
                        })
                        r2_perm_tensors.append(r2_perm_stage1)

                        if r2_max_t > best_tensor:
                            best_tensor = r2_max_t
                            best_idx = j

                    r2_perm_max = torch.stack(r2_perm_tensors, dim=0).amax(dim=0)
                    best_info = per_pheno[best_idx] if best_idx >= 0 else None
                    pval_perm = float("inf")
                    pval_beta = float("nan")
                    a_hat = float("nan")
                    b_hat = float("nan")
                    p_true = float("nan")
                    true_dof_val = float("nan")
                    stop_pval = float("inf")

                    if best_info is not None:
                        pval_perm = (
                            (r2_perm_max >= best_info["tensor"]).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                        ).item()
                        if beta_approx:
                            r2_perm_np = r2_perm_max.detach().cpu().numpy()
                            pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                                r2_perm_np, float(best_info["tensor"].item()), dof_init=best_info["dof"],
                            )
                        else:
                            pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                        stop_pval = float(pval_beta)
                        if not np.isfinite(stop_pval):
                            stop_pval = pval_perm

                    if best_info is not None and stage_cap < nperm and stop_pval <= signif_threshold * 2:
                        remaining = nperm - r2_perm_max.numel()
                        if remaining > 0:
                            updated_vectors: list[torch.Tensor] = []
                            for info in per_pheno:
                                _, _, _, _, extra_perm = compute_perm_r2_max(
                                    y_resid=y_resid_list[ids_list.index(info["pid"])],
                                    G_resid=I_t,
                                    H_resid=H_cov,
                                    k_eff=k_eff,
                                    perm_stream=info["stream"],
                                    max_permutations=remaining,
                                    return_nominal=False,
                                    mixed_precision=mixed_precision,
                                )
                                if extra_perm.numel():
                                    info["r2_perm"] = torch.cat([info["r2_perm"], extra_perm])
                                updated_vectors.append(info["r2_perm"])
                            if updated_vectors:
                                r2_perm_max = torch.stack(updated_vectors, dim=0).amax(dim=0)
                                pval_perm = (
                                    (r2_perm_max >= best_info["tensor"]).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                                ).item()
                                if beta_approx:
                                    r2_perm_np = r2_perm_max.detach().cpu().numpy()
                                    pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                                        r2_perm_np, float(best_info["tensor"].item()), dof_init=best_info["dof"],
                                    )
                                else:
                                    pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                                stop_pval = float(pval_beta)
                                if not np.isfinite(stop_pval):
                                    stop_pval = pval_perm

                    if best_info is None:
                        return {
                            "best_ix_var": -1,
                            "best_pheno_idx": -1,
                            "stop_pval": float("inf"),
                        }

                    best_pid = per_pheno[best_idx]["pid"]
                    best_ix = per_pheno[best_idx]["ix"]
                    best_beta = per_pheno[best_idx]["beta"]
                    best_se = per_pheno[best_idx]["se"]
                    best_t = per_pheno[best_idx]["tval"]
                    best_dof = per_pheno[best_idx]["dof"]
                    best_r2 = float(per_pheno[best_idx]["tensor"].item())
                    pval_nominal = float(get_t_pval(best_t, best_dof))

                    true_dof_out = int(true_dof_val) if np.isfinite(true_dof_val) else int(best_dof)

                    return {
                        "best_ix_var": int(best_ix),
                        "best_pheno_idx": int(best_idx),
                        "best_pid": best_pid,
                        "beta": best_beta,
                        "se": best_se,
                        "tval": best_t,
                        "r2_nom": best_r2,
                        "pval_nominal": pval_nominal,
                        "pval_perm": float(pval_perm),
                        "pval_beta": float(pval_beta),
                        "a_hat": float(a_hat),
                        "b_hat": float(b_hat),
                        "p_true": float(p_true),
                        "true_dof": true_dof_out,
                        "stop_pval": float(stop_pval),
                        "num_var": num_var,
                        "dof": best_dof,
                    }

                per_pheno: list[dict[str, object]] = []
                r2_perm_tensors: list[torch.Tensor] = []
                best_tensor = torch.tensor(float("-inf"), device=G_resid.device, dtype=torch.float32)
                best_idx = -1

                p_pred = 1 + (H_resid.shape[2] if H_resid is not None else 0)

                for j, (pid_inner, y_resid) in enumerate(zip(ids_list, y_resid_list)):
                    stream = PermutationStream(
                        n_samples=int(y_resid.shape[0]),
                        nperm=nperm,
                        device=y_resid.device,
                        chunk_size=max(1, perm_chunk_local),
                        seed=perm_seed_base,
                        mode=perm_indices_mode,
                    )
                    betas, ses, tstats, r2_nom_vec, r2_perm_stage1 = compute_perm_r2_max(
                        y_resid=y_resid,
                        G_resid=G_resid,
                        H_resid=H_resid,
                        k_eff=k_eff,
                        perm_stream=stream,
                        max_permutations=stage_cap,
                        return_nominal=True,
                        mixed_precision=mixed_precision,
                    )
                    if r2_nom_vec is None:
                        r2_nom_vec = torch.zeros(num_var, device=y_resid.device, dtype=torch.float32)
                    r2_nom_vec = torch.nan_to_num(r2_nom_vec.to(torch.float32), nan=-1.0)
                    r2_max_t, ix_t = _nanmax(r2_nom_vec, dim=0)
                    ix = int(ix_t.item())
                    if random_tiebreak:
                        ties = torch.nonzero(torch.isclose(r2_nom_vec, r2_max_t, atol=1e-12), as_tuple=True)[0]
                        if ties.numel() > 1:
                            if gen is None:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device)
                            else:
                                choice_tensor = torch.randint(0, ties.numel(), (1,), device=ties.device, generator=gen)
                            ix = int(ties[int(choice_tensor.item())].item())
                            r2_max_t = r2_nom_vec[ix]

                    dof = max(int(y_resid.shape[0]) - int(k_eff) - int(p_pred), 1)
                    beta_val = float(betas[ix, 0].item())
                    se_val = float(ses[ix, 0].item())
                    tval = float(tstats[ix, 0].item())

                    per_pheno.append({
                        "pid": pid_inner,
                        "stream": stream,
                        "betas": betas,
                        "ses": ses,
                        "tstats": tstats,
                        "r2_nominal_vec": r2_nom_vec,
                        "ix": ix,
                        "tensor": r2_max_t,
                        "beta": beta_val,
                        "se": se_val,
                        "tval": tval,
                        "dof": dof,
                        "r2_perm": r2_perm_stage1,
                    })
                    r2_perm_tensors.append(r2_perm_stage1)

                    if r2_max_t > best_tensor:
                        best_tensor = r2_max_t
                        best_idx = j

                r2_perm_max = torch.stack(r2_perm_tensors, dim=0).amax(dim=0)
                best_info = per_pheno[best_idx] if best_idx >= 0 else None
                pval_perm = float("inf")
                pval_beta = float("nan")
                a_hat = float("nan")
                b_hat = float("nan")
                p_true = float("nan")
                true_dof_val = float("nan")
                stop_pval = float("inf")

                if best_info is not None:
                    pval_perm = (
                        (r2_perm_max >= best_info["tensor"]).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                    ).item()
                    if beta_approx:
                        r2_perm_np = r2_perm_max.detach().cpu().numpy()
                        pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                            r2_perm_np, float(best_info["tensor"].item()), dof_init=best_info["dof"],
                        )
                    else:
                        pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                    stop_pval = float(pval_beta)
                    if not np.isfinite(stop_pval):
                        stop_pval = pval_perm

                if best_info is not None and stage_cap < nperm and stop_pval <= signif_threshold * 2:
                    remaining = nperm - r2_perm_max.numel()
                    if remaining > 0:
                        updated_vectors: list[torch.Tensor] = []
                        for info in per_pheno:
                            _, _, _, _, extra_perm = compute_perm_r2_max(
                                y_resid=y_resid_list[ids_list.index(info["pid"])],
                                G_resid=G_resid,
                                H_resid=H_resid,
                                k_eff=k_eff,
                                perm_stream=info["stream"],
                                max_permutations=remaining,
                                return_nominal=False,
                                mixed_precision=mixed_precision,
                            )
                            if extra_perm.numel():
                                info["r2_perm"] = torch.cat([info["r2_perm"], extra_perm])
                            updated_vectors.append(info["r2_perm"])
                        if updated_vectors:
                            r2_perm_max = torch.stack(updated_vectors, dim=0).amax(dim=0)
                            pval_perm = (
                                (r2_perm_max >= best_info["tensor"]).sum().add_(1).float() / (r2_perm_max.numel() + 1)
                            ).item()
                            if beta_approx:
                                r2_perm_np = r2_perm_max.detach().cpu().numpy()
                                pval_beta, a_hat, b_hat, true_dof_val, p_true = beta_approx_pval(
                                    r2_perm_np, float(best_info["tensor"].item()), dof_init=best_info["dof"],
                                )
                            else:
                                pval_beta = a_hat = b_hat = true_dof_val = p_true = np.nan
                            stop_pval = float(pval_beta)
                            if not np.isfinite(stop_pval):
                                stop_pval = pval_perm

                if best_info is None:
                    return {
                        "best_ix_var": -1,
                        "best_pheno_idx": -1,
                        "stop_pval": float("inf"),
                    }

                best_pid = per_pheno[best_idx]["pid"]
                best_ix = per_pheno[best_idx]["ix"]
                best_beta = per_pheno[best_idx]["beta"]
                best_se = per_pheno[best_idx]["se"]
                best_t = per_pheno[best_idx]["tval"]
                best_dof = per_pheno[best_idx]["dof"]
                best_r2 = float(per_pheno[best_idx]["tensor"].item())
                pval_nominal = float(get_t_pval(best_t, best_dof))

                true_dof_out = int(true_dof_val) if np.isfinite(true_dof_val) else int(best_dof)

                return {
                    "best_ix_var": int(best_ix),
                    "best_pheno_idx": int(best_idx),
                    "best_pid": best_pid,
                    "beta": best_beta,
                    "se": best_se,
                    "tval": best_t,
                    "r2_nom": best_r2,
                    "pval_nominal": pval_nominal,
                    "pval_perm": float(pval_perm),
                    "pval_beta": float(pval_beta),
                    "a_hat": float(a_hat),
                    "b_hat": float(b_hat),
                    "p_true": float(p_true),
                    "true_dof": true_dof_out,
                    "stop_pval": float(stop_pval),
                    "num_var": num_var,
                    "dof": best_dof,
                }

            forward_records: list[dict[str, object]] = []
            base_record = {col: seed_row.get(col) for col in expected_columns if col not in ("rank", "group_size")}
            base_record["group_size"] = len(ids_list)
            base_record["group_id"] = group_id
            forward_records.append(base_record)

            while True:
                extras_ids = list(dosage_proj.keys())
                eval_res = evaluate_group_model(extras_ids)
                if eval_res["best_ix_var"] < 0 or eval_res["stop_pval"] > signif_threshold:
                    break

                best_ix_var = eval_res["best_ix_var"]
                best_ix_pheno = eval_res["best_pheno_idx"]
                pid_best = ids_list[best_ix_pheno]
                var_id = idx_to_id[v_idx[best_ix_var]]
                var_pos = int(pos_arr[v_idx[best_ix_var]])
                start_pos = ig.phenotype_start[pid_best]
                end_pos = ig.phenotype_end[pid_best]

                g_sel = G_t[best_ix_var]
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

                forward_records.append({
                    "group_id": group_id,
                    "group_size": len(ids_list),
                    "phenotype_id": pid_best,
                    "variant_id": var_id,
                    "start_distance": int(var_pos - start_pos),
                    "end_distance": int(var_pos - end_pos),
                    "num_var": eval_res["num_var"],
                    "slope": eval_res["beta"],
                    "slope_se": eval_res["se"],
                    "tstat": eval_res["tval"],
                    "r2_nominal": eval_res["r2_nom"],
                    "pval_nominal": eval_res["pval_nominal"],
                    "pval_perm": eval_res["pval_perm"],
                    "pval_beta": eval_res["pval_beta"],
                    "beta_shape1": eval_res["a_hat"],
                    "beta_shape2": eval_res["b_hat"],
                    "true_dof": eval_res["true_dof"],
                    "pval_true_dof": eval_res["p_true"],
                    "af": af,
                    "ma_samples": ma_samples,
                    "ma_count": ma_count,
                })
                if interaction_mode and interaction_columns:
                    for anc_idx in range(len(interaction_columns) // 4):
                        base = anc_idx * 4
                        forward_records[-1][interaction_columns[base + 0]] = eval_res["slope_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 1]] = eval_res["slope_se_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 2]] = eval_res["tstat_gxh"][anc_idx]
                        forward_records[-1][interaction_columns[base + 3]] = eval_res["pval_gxh"][anc_idx]

                if var_id in var_in_frame and var_id in geno_has_variant and var_id not in dosage_proj:
                    ensure_projected(var_id)

            if not forward_records:
                continue

            if len(forward_records) > 1:
                kept_records: list[dict[str, object]] = []
                selected = [rec["variant_id"] for rec in forward_records]

                for rk, drop_vid in enumerate(selected, start=1):
                    kept = [v for v in selected if v != drop_vid]
                    eval_res = evaluate_group_model(kept)
                    if eval_res["best_ix_var"] >= 0 and eval_res["stop_pval"] <= signif_threshold:
                        pid_best = ids_list[eval_res["best_pheno_idx"]]
                        var_id = idx_to_id[v_idx[eval_res["best_ix_var"]]]
                        var_pos = int(pos_arr[v_idx[eval_res["best_ix_var"]]])
                        start_pos = ig.phenotype_start[pid_best]
                        end_pos = ig.phenotype_end[pid_best]

                        g_sel = G_t[eval_res["best_ix_var"]]
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

                        kept_records.append({
                            "group_id": group_id,
                            "group_size": len(ids_list),
                            "phenotype_id": pid_best,
                            "variant_id": var_id,
                            "start_distance": int(var_pos - start_pos),
                            "end_distance": int(var_pos - end_pos),
                            "num_var": eval_res["num_var"],
                            "slope": eval_res["beta"],
                            "slope_se": eval_res["se"],
                            "tstat": eval_res["tval"],
                            "r2_nominal": eval_res["r2_nom"],
                            "pval_nominal": eval_res["pval_nominal"],
                            "pval_perm": eval_res["pval_perm"],
                            "pval_beta": eval_res["pval_beta"],
                            "beta_shape1": eval_res["a_hat"],
                            "beta_shape2": eval_res["b_hat"],
                            "true_dof": eval_res["true_dof"],
                            "pval_true_dof": eval_res["p_true"],
                            "af": af,
                            "ma_samples": ma_samples,
                            "ma_count": ma_count,
                            "rank": int(rk),
                        })
                        if interaction_mode and interaction_columns:
                            for anc_idx in range(len(interaction_columns) // 4):
                                base = anc_idx * 4
                                kept_records[-1][interaction_columns[base + 0]] = eval_res["slope_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 1]] = eval_res["slope_se_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 2]] = eval_res["tstat_gxh"][anc_idx]
                                kept_records[-1][interaction_columns[base + 3]] = eval_res["pval_gxh"][anc_idx]

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
        tensorqtl_flavor: bool = False, perm_indices_mode: str = "generator",
        mixed_precision: str | None = None, nperm_stage1: int | None = 2000,
        ancestry_model: str = "main",
        interaction_covariate: Optional[object] = None,
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
    if ancestry_model == "interaction":
        logger.write("  * ancestry model: interaction (GxH)")
    if interaction_covariate is not None:
        logger.write("  * covariate interaction: GxC")

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

    if group_s is None:
        if "phenotype_id" not in signif_df.columns:
            raise ValueError("cis_df must contain 'phenotype_id' for ungrouped mapping.")
        signif_seed_df = signif_df.set_index("phenotype_id", drop=False)
        valid_ids = ig.phenotype_pos_df.index.intersection(signif_seed_df.index)
        phenotype_counts = ig.phenotype_pos_df.loc[valid_ids, "chr"].value_counts().to_dict()
        total_items = int(valid_ids.shape[0])
        item_label = "phenotypes"

        n_anc = int(haplotypes.shape[2]) if haplotypes is not None else None
        def run_core(chrom: str | None, chrom_total: int | None) -> pd.DataFrame:
            return _run_independent_core(
                ig=ig, variant_df=variant_df, covariates_df=covariates_use,
                signif_seed_df=signif_seed_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
                perm_chunk=perm_chunk,
                logger=logger, total_items=chrom_total, item_label=item_label,
                tensorqtl_flavor=tensorqtl_flavor,
                perm_indices_mode=perm_indices_mode,
                mixed_precision=mixed_precision,
                nperm_stage1=nperm_stage1,
                ancestry_model=ancestry_model,
                n_ancestries=n_anc,
                interaction_covariate_t=interaction_cov_t,
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

        n_anc = int(haplotypes.shape[2]) if haplotypes is not None else None
        def run_core(chrom: str | None, chrom_total: int | None) -> pd.DataFrame:
            return _run_independent_core_group(
                ig=ig, variant_df=variant_df, covariates_df=covariates_use,
                seed_by_group_df=seed_by_group_df, signif_threshold=signif_threshold,
                nperm=nperm, device=device, maf_threshold=maf_threshold,
                random_tiebreak=random_tiebreak, missing=missing,
                beta_approx=beta_approx, seed=seed, chrom=chrom,
                perm_chunk=perm_chunk,
                logger=logger, total_items=chrom_total, item_label=item_label,
                tensorqtl_flavor=tensorqtl_flavor,
                perm_indices_mode=perm_indices_mode,
                mixed_precision=mixed_precision,
                nperm_stage1=nperm_stage1,
                ancestry_model=ancestry_model,
                n_ancestries=n_anc,
                interaction_covariate_t=interaction_cov_t,
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
                logger.write(f"    Chromosome {chrom} completed in {elapsed / 60:.2f} min")

    if logger.verbose:
        elapsed = time.time() - overall_start
        logger.write(f"    Completed independent scan in {elapsed / 60:.2f} min")

    if results:
        return pd.concat(results, axis=0, ignore_index=True)
    return pd.DataFrame()
