import numpy as np
import torch

from src.localqtl.cis._permute import compute_perm_r2_max
from src.localqtl.cis.permutations import map_permutations
from src.localqtl.regression_kernels import (
    run_batch_regression,
    run_batch_regression_with_permutations,
)
from src.localqtl.utils import SimpleLogger


def test_map_permutations_computes_empirical_stats(toy_data):
    torch.manual_seed(0)
    result = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=5,
        seed=123,
        logger=SimpleLogger(verbose=False),
    )

    assert {"phenotype_id", "variant_id", "pval_perm", "pval_beta", "true_dof"}.issubset(result.columns)
    expected_ids = {pid for pid in toy_data["phenotype_df"].index if pid != "geneConst"}
    assert set(result["phenotype_id"]) == expected_ids
    assert ((result["pval_perm"] >= 0.0) & (result["pval_perm"] <= 1.0)).all()
    assert np.isfinite(result[["slope", "slope_se", "tstat", "r2_nominal"]]).all().all()


def test_perm_partial_matches_nominal_when_H_none():
    torch.manual_seed(0)
    n, m = 12, 5
    y = torch.randn(n)
    G = torch.randn(m, n)
    y_perm = torch.randn(n, 7)

    betas, ses, tstats, r2_perm = run_batch_regression_with_permutations(
        y=y, G=G, H=None, y_perm=y_perm, k_eff=0, device="cpu"
    )

    EPS = 1e-8
    Gnorm2 = (G * G).sum(dim=1) + EPS
    ynorm2 = (y * y).sum() + EPS
    Gy = G @ y
    dof = max(n - 1, 1)
    r = Gy / torch.sqrt(Gnorm2 * ynorm2)
    t = r * torch.sqrt(dof / torch.clamp(1.0 - r * r, min=EPS))
    beta_expected = Gy / Gnorm2
    se_expected = torch.abs(beta_expected) / torch.clamp(t.float(), min=EPS)

    assert torch.allclose(betas.squeeze(1), beta_expected, atol=1e-6)
    assert torch.allclose(ses.squeeze(1), se_expected, atol=1e-6)
    assert torch.allclose(tstats.squeeze(1), t, atol=1e-6)

    Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS
    GYp = G @ y_perm
    denom = torch.sqrt(Gnorm2).unsqueeze(1) * torch.sqrt(Ypnorm2).unsqueeze(0)
    R2 = (GYp / torch.clamp(denom, min=EPS)) ** 2
    r2_expected = R2.max(dim=0).values.float()
    assert torch.allclose(r2_perm, r2_expected, atol=1e-6)


def test_perm_uses_H_for_partial_statistic():
    torch.manual_seed(1)
    n, m, pH = 10, 4, 2
    base_H = torch.randn(n, pH)
    H = base_H.unsqueeze(0).expand(m, -1, -1).contiguous()
    y = base_H[:, 0] + 0.05 * torch.randn(n)
    G = torch.randn(m, n)
    y_perm = torch.randn(n, 6)

    _, _, _, r2_partial = run_batch_regression_with_permutations(
        y=y, G=G, H=H, y_perm=y_perm, k_eff=0, device="cpu"
    )
    _, _, _, r2_correlation = run_batch_regression_with_permutations(
        y=y, G=G, H=H, y_perm=y_perm, k_eff=0, device="cpu", use_partial_perm=False
    )

    assert not torch.allclose(r2_partial, r2_correlation, atol=1e-6)


def test_dof_consistency_between_nominal_and_permutation():
    torch.manual_seed(2)
    n, m, pH = 9, 1, 2
    k_eff = 1
    y = torch.randn(n)
    G = torch.randn(m, n)
    H = torch.randn(m, n, pH)
    perm_ix = torch.stack([torch.randperm(n) for _ in range(4)], dim=0)

    betas, ses, tstats, r2_perm = compute_perm_r2_max(
        y_resid=y,
        G_resid=G,
        H_resid=H,
        k_eff=k_eff,
        perm_ix=perm_ix,
        device="cpu",
        perm_chunk=2,
        return_nominal=True,
    )

    assert betas is not None and ses is not None and tstats is not None

    p = 1 + pH
    dof_expected = max(n - k_eff - p, 1)

    betas_run, ses_run, tstats_run = run_batch_regression(
        y=y, G=G, H=H, k_eff=k_eff, device="cpu"
    )
    assert torch.allclose(betas, betas_run)
    assert torch.allclose(ses, ses_run)
    assert torch.allclose(tstats, tstats_run)

    r2_from_t = (
        tstats[:, 0].double().pow(2) / (tstats[:, 0].double().pow(2) + dof_expected)
    ).to(torch.float32)
    r2_from_run = (
        tstats_run[:, 0].double().pow(2) / (tstats_run[:, 0].double().pow(2) + dof_expected)
    ).to(torch.float32)
    assert torch.allclose(r2_from_t, r2_from_run, atol=1e-6)

    for i, perm in enumerate(perm_ix):
        y_perm_vec = y.index_select(0, perm)
        betas_perm, ses_perm, tstats_perm = run_batch_regression(
            y=y_perm_vec, G=G, H=H, k_eff=k_eff, device="cpu"
        )
        t_sq = tstats_perm[:, 0].double().pow(2)
        r2_perm_manual = (t_sq / (t_sq + dof_expected)).to(torch.float32)
        r2_perm_max_manual = torch.nan_to_num(r2_perm_manual, nan=-1.0).max()
        assert torch.allclose(r2_perm[i], r2_perm_max_manual, atol=1e-5)


def test_map_permutations_with_haplotypes(toy_hap_data_3anc):
    torch.manual_seed(0)
    result = map_permutations(
        genotype_df=toy_hap_data_3anc["genotype_df"],
        variant_df=toy_hap_data_3anc["variant_df"],
        phenotype_df=toy_hap_data_3anc["phenotype_df"],
        phenotype_pos_df=toy_hap_data_3anc["phenotype_pos_df"],
        covariates_df=None,
        haplotypes=toy_hap_data_3anc["haplotypes"],
        loci_df=toy_hap_data_3anc["loci_df"],
        device="cpu",
        nperm=4,
        seed=321,
        logger=SimpleLogger(verbose=False),
    )

    assert not result.empty
    expected_cols = {
        "phenotype_id", "variant_id", "pval_perm", "pval_beta", "true_dof"
    }
    assert expected_cols.issubset(result.columns)
