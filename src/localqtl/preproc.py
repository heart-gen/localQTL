## Adapted from tensorqtl
## https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
import numpy as np
import torch

__all__ = [
    "impute_mean_and_filter",
    "calculate_maf",
    "allele_stats",
    "filter_by_maf",
    "ancestry_allele_stats",
    "filter_maf_interaction",
    "interaction_half_mask",
]

def impute_mean_and_filter(
        G_t: torch.Tensor, missing: float = -9.0,
        allow_nonfinite: bool = True, inplace: bool = False,
):
    """
    Mean-impute per variant and mark monomorphic rows.
    """
    if not inplace:
        G_t = G_t.clone()

    miss_mask = (G_t == missing)
    if allow_nonfinite:
        miss_mask = miss_mask | (~torch.isfinite(G_t))

    # Mean-impute first, then drop rows that are monomorphic *after* imputation
    # (a row such as [-9, 1, 1, 1] is constant once imputed and would otherwise
    # produce an all-zero centred predictor).
    n_obs = (~miss_mask).sum(dim=1, keepdim=True)
    sum_obs = torch.where(miss_mask, torch.zeros_like(G_t), G_t).sum(dim=1, keepdim=True)
    mu = sum_obs / n_obs.clamp_min(1)
    G_imputed = torch.where(miss_mask, mu, G_t)

    mono = (G_imputed == G_imputed[:, [0]]).all(dim=1)
    keep_mask = ~mono

    if keep_mask.sum().item() == 0:
        return G_t.new_empty((0, G_t.shape[1])), keep_mask, miss_mask

    return G_imputed[keep_mask], keep_mask, miss_mask


def calculate_maf(G_t: torch.Tensor, *, ploidy: int = 2) -> torch.Tensor:
    """
    MAF per variant assuming dosage-coded genotypes in [0, ploidy].
    Expects missing already imputed; use with impute_mean_and_filter() first.
    """
    af = G_t.sum(dim=1) / (ploidy * G_t.shape[1])
    return torch.where(af > 0.5, 1 - af, af)


def allele_stats(G_t: torch.Tensor, *, ploidy: int = 2):
    """
    Allele frequency, #minor-allele carriers, and minor-allele count per variant.
    Assumes missing already imputed. Suitable for dosages.
    """
    n2 = ploidy * G_t.shape[1]
    af_t = G_t.sum(dim=1) / n2

    # carriers under the "minor-allele side" (thresholds are dosage-friendly)
    is_minor = af_t <= 0.5
    has_minor = (G_t > 0.5)     # carriers of alt allele under AF<=0.5
    has_major = (G_t < 1.5)     # carriers of ref allele under AF>0.5

    ma_samples = torch.where(is_minor, has_minor.sum(1), has_major.sum(1)).to(torch.int32)
    minor_count = (G_t * has_minor.float()).sum(1)         # alt dosage when AF<=0.5
    ma_count = torch.where(is_minor, minor_count, n2 - minor_count).to(torch.int32)
    return af_t, ma_samples, ma_count


def filter_by_maf(G_t: torch.Tensor, maf_threshold: float, *, ploidy: int = 2):
    """
    Returns (keep_mask, maf_t). Assumes missing already imputed.
    """
    if maf_threshold <= 0:
        maf = calculate_maf(G_t, ploidy=ploidy)
        return torch.ones(G_t.shape[0], dtype=torch.bool, device=G_t.device), maf
    maf = calculate_maf(G_t, ploidy=ploidy)
    keep = maf >= maf_threshold
    return keep, maf


def ancestry_allele_stats(G_t: torch.Tensor, H_t: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Per-ancestry in-sample allele statistics for the ancestry-specific dosage model.

    ``G_t`` is ``(m, n)`` imputed dosages of the counted (alt) allele and ``H_t``
    ``(m, n, K)`` (expected) ancestry counts per sample; a single channel is
    expanded to ``[H, 2 - H]`` as in :func:`localqtl.cis.common.ancestry_dosage_design`.
    With the phase unknown the expected alt-allele count carried on ancestry-``k``
    haplotypes is ``G * L_k / 2`` (exact for hard calls).

    Returns ``an`` (allele number), ``ac`` (counted-allele count), ``af``
    (``ac / an``, NaN where ``an == 0``) and ``ma_count`` (minor-allele count
    folded within the ancestry), each ``(m, K)`` float32. No per-ancestry
    carrier count is reported: without phase a heterozygote with one
    ancestry-``k`` haplotype cannot be assigned to either ancestry.
    """
    if H_t.dim() != 3:
        raise ValueError("H_t must be (variants, samples, channels).")
    H_t = H_t.to(dtype=torch.float32)
    L = torch.cat([H_t, 2.0 - H_t], dim=2) if H_t.shape[2] == 1 else H_t
    an = L.sum(dim=1)                                        # (m, K)
    ac = (G_t.to(torch.float32).unsqueeze(-1) * L * 0.5).sum(dim=1)
    af = torch.where(an > 0, ac / an.clamp_min(1e-12), torch.full_like(an, float("nan")))
    ma_count = torch.minimum(ac, an - ac)
    return {"an": an, "ac": ac, "af": af, "ma_count": ma_count}


def interaction_half_mask(interaction: "torch.Tensor | np.ndarray") -> torch.Tensor:
    """
    tensorQTL's split of the samples into the lower and upper half of a
    (single) interaction variable: ``True`` marks the upper half (stable sort,
    ties resolved by sample order).  Used by :func:`filter_maf_interaction`.
    """
    vals = interaction.detach().cpu().numpy() if torch.is_tensor(interaction) else np.asarray(interaction)
    vals = vals.reshape(-1)
    order = np.argsort(vals, kind="mergesort")
    mask = np.ones(vals.shape[0], dtype=bool)
    mask[order[: vals.shape[0] // 2]] = False
    return torch.as_tensor(mask)


def filter_maf_interaction(
        G_t: torch.Tensor,
        interaction_mask_t: torch.Tensor | None = None,
        maf_threshold_interaction: float = 0.05,
):
    """
    Variant filter for interaction scans (tensorQTL ``filter_maf_interaction``):
    drops monomorphic sites (all 0, all 1 or all 2) and, when
    ``interaction_mask_t`` (``True`` = upper half of the interaction variable)
    is given, keeps only variants with MAF >= ``maf_threshold_interaction`` in
    *both* halves of the samples, so that the interaction is estimable in each.

    Returns ``(G_kept, keep_mask)``.
    """
    mask = ~((G_t == 0).all(1) | (G_t == 1).all(1) | (G_t == 2).all(1))
    if interaction_mask_t is not None and maf_threshold_interaction and maf_threshold_interaction > 0:
        im = interaction_mask_t.to(device=G_t.device, dtype=torch.bool)
        upper = calculate_maf(G_t[:, im]) >= maf_threshold_interaction - 1e-7
        lower = calculate_maf(G_t[:, ~im]) >= maf_threshold_interaction - 1e-7
        mask = mask & upper & lower
    return G_t[mask], mask
