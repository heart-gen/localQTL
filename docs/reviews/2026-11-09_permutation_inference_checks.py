"""Numeric checks behind docs/reviews/2026-11-09_permutation_inference.md.

    python docs/reviews/2026-11-09_permutation_inference_checks.py [--draws 40000] [--threads 4]

Everything is drawn here (no data files): a cohort of 207 samples and 24 covariates, a window of 400
variants in LD and a two-ancestry local-ancestry panel, i.e. the shape of one BrainSeq caudate window.
The phenotypes are exact Gaussian nulls in the covariate-orthogonal space, so every statistic below has
a known reference.  CPU, float32, a few seconds.

(A) the observed scan and the permutation null compute one statistic on the same vector;
(B) the Beta approximation of the dosage scan against that exact null, for both lead rules, with the
    nominal denominator dof (the package's fit) and with an effective dof that sets the moment shape1 to 1.
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.optimize as so
import torch
from scipy import stats as st

from localqtl import regression_kernels as rk
from localqtl.cis.common import residualize_batch, residualize_batch_dosage
from localqtl.cis.permutations import _pooled_lead_observed
from localqtl.stats import _beta_mle_on_p, get_f_pval


def design(seed: int = 1, n: int = 207, k: int = 24, m: int = 400):
    """Genotypes in LD (AR(1) latent, rho .95), haplotype ancestry 80 % ancestry 0 with rare switches along
    the window, ancestry-specific allele frequencies, Gaussian covariates."""
    rng = np.random.default_rng(seed)
    z = np.empty((m, 2 * n))
    z[0] = rng.normal(size=2 * n)
    for j in range(1, m):
        z[j] = 0.95 * z[j - 1] + np.sqrt(1 - 0.95 ** 2) * rng.normal(size=2 * n)
    anc = rng.random(2 * n) < 0.8
    switch = rng.random((m, 2 * n)) < 0.002
    anc_m = np.logical_xor(anc[None, :], np.cumsum(switch, axis=0) % 2 == 1)
    maf = rng.uniform(0.05, 0.5, size=m)[:, None]
    maf_other = np.clip(maf * rng.uniform(0.3, 1.7, size=(m, 1)), 0.02, 0.6)
    hap = (z < st.norm.ppf(np.where(anc_m, maf, maf_other))).astype(np.float32)
    G = torch.tensor(hap[:, :n] + hap[:, n:])
    H = torch.tensor((anc_m[:, :n].astype(np.float32) + anc_m[:, n:])[:, :, None])     # ancestry-0 haplotype count
    C = torch.tensor(rng.normal(size=(n, k)).astype(np.float32))
    return G, H, C


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=40_000, help="null phenotypes (half fit the Beta, half score it)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    G, H, C = design(args.seed)
    m, n = G.shape
    rez = rk.Residualizer(C)
    y = torch.randn(n)
    y_res, D, L = residualize_batch_dosage(y, G, H, rez, center=True, group=False)
    G_sum = D.sum(dim=2)
    ctx = rk.prep_ctx_for_dosage_perm(D, L, k_eff=rez.k_eff)
    K, dof = int(ctx["K"]), int(ctx["dof"])
    print(f"design: n {n}, covariates {C.shape[1]}, variants {m}, K {K}, k_eff {rez.k_eff}, joint dof {dof}")

    # ---------------------------------------------------------------- (A) one statistic, observed and null
    ctx_p, r2_obs, *_ = _pooled_lead_observed(y_res, G_sum, L, rez.k_eff, None)
    r2_null = rk._perm_chunk_r2_block(ctx_p, L, G_sum, y_res.reshape(-1, 1))[:, 0]
    print(f"(A1) pooled R2, observed path vs null path: max abs diff {float((r2_obs - r2_null).abs().max()):.2e}, "
          f"same lead {int(r2_obs.argmax()) == int(r2_null.argmax())}")
    f_block = rk.dosage_fstat(ctx, y_res)
    f_refit = rk.run_batch_dosage_regression(y_res, D, L, k_eff=rez.k_eff, device="cpu")["fstat"]
    print(f"(A2) joint F, block estimator vs refit: max rel diff "
          f"{float(((f_block - f_refit).abs() / f_refit.clamp_min(1e-6)).max()):.2e}")
    y_m, G_m, H_m = residualize_batch(y, G, H, rez, center=True)
    _, _, ts = rk.run_batch_regression(y_m, G_m, H_m, k_eff=rez.k_eff, device="cpu")
    t2 = ts[:, 0].double().pow(2)
    r2_main = (t2 / (t2 + (n - rez.k_eff - 2))).float()
    ctx_m, *_ = rk.prep_ctx_for_perm(y_m, G_m, H_m, rez.k_eff)
    r2_main_null = rk._perm_chunk_r2_block(ctx_m, H_m, G_m, y_m.reshape(-1, 1))[:, 0]
    print(f"(A3) main model R2, ridge OLS (observed) vs partial-correlation kernel (null): max abs diff "
          f"{float((r2_main - r2_main_null).abs().max()):.2e}, max rel diff "
          f"{float(((r2_main - r2_main_null).abs() / r2_main_null.clamp_min(1e-6)).max()):.2e}")

    # ---------------------------------------------------------------- (B) the Beta fit against an exact null
    N, h = int(args.draws), int(args.draws) // 2
    Y = rez.transform(torch.randn(N, n), center=True)[0].T.contiguous()           # (n, N), exact Gaussian null
    F = torch.nan_to_num(rk._dosage_fstat_block(ctx, Y), nan=-1.0)
    ix = rk._perm_chunk_r2_block(ctx_p, L, G_sum, Y).argmax(dim=0)
    nulls = {"joint lead: maximum F over the window": F.amax(dim=0).numpy().astype(np.float64),
             "pooled lead: F at the pooled-selected variant": F.gather(0, ix[None, :])[0].numpy().astype(np.float64)}

    def moment_shape1(f, dfd):
        p = get_f_pval(f, K, dfd)
        mu, v = p.mean(), p.var()
        return mu * (mu * (1 - mu) / v - 1)

    def report(tag, u):
        ks = st.kstest(u, "uniform")
        print(f"     {tag}: KS D {ks.statistic:.4f} (p {ks.pvalue:.2g}); below .05 / .01 / .001: "
              f"{np.mean(u < .05):.4f} / {np.mean(u < .01):.4f} / {np.mean(u < .001):.4f}")

    print(f"(B) Beta approximation on {N} exact null phenotypes: fitted on {h}, scored on the other {N - h}")
    for name, f in nulls.items():
        fit, held = f[:h], f[h:]
        print(f"   {name}")
        a, b = _beta_mle_on_p(get_f_pval(fit, K, dof))
        report(f"package fit, nominal dfd {dof}, shape1 {a:.3f}, shape2 {b:.1f}",
               st.beta.cdf(get_f_pval(held, K, dof), a, b))
        log_dfd = so.brentq(lambda x: np.log(moment_shape1(fit, np.exp(x))), np.log(dof / 20), np.log(dof * 20))
        dfd = float(np.exp(log_dfd))
        a2, b2 = _beta_mle_on_p(get_f_pval(fit, K, dfd))
        report(f"effective dfd {dfd:.1f} (moment shape1 = 1), shape1 {a2:.3f}",
               st.beta.cdf(get_f_pval(held, K, dfd), a2, b2))
        rank = np.searchsorted(np.sort(fit), held, side="left")
        report("empirical p against the fitting half (the pval_perm construction)", ((h - rank) + 1) / (h + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
