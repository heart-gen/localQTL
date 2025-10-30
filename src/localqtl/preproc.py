## Adapted from tensorqtl
## https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
import torch

__all__ = [
    "impute_mean_and_filter",
    "calculate_maf",
    "allele_stats",
    "filter_by_maf",
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

    miss = (G_t == missing)
    if allow_nonfinite:
        miss = miss | (~torch.isfinite(G_t))

    # counts / sums over observed entries
    n_obs = (~miss).sum(dim=1, keepdim=True)
    sum_obs = torch.where(miss, torch.zeros_like(G_t), G_t).sum(dim=1, keepdim=True)

    # avoid div-by-zero; rows with n_obs==0 will be flagged below
    mu = sum_obs / n_obs.clamp_min(1)
    G_t = torch.where(miss, mu, G_t)

    # mark rows with no observed data OR zero variance as not-keep
    all_missing = (n_obs.squeeze(1) == 0)
    keep = (~all_missing) & (G_t.var(dim=1, unbiased=False) > 0)

    return G_t, keep, miss


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
