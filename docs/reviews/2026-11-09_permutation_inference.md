# Review of permutation inference, 2026-11-09

Reviewed: localQTL 0.4.4.dev1 at `76ba640` (branch `feat/pooled-lead-selection`), CPU, float32.

| Priority | Code | Objective |
|---|---|---|
| High | `src/localqtl/cis/permutations.py` | Observed and permuted pooled-lead selection use identical procedures, including tie handling and covariate adjustment |
| High | `src/localqtl/regression_kernels.py` | Joint F: calculation, degrees of freedom, numerical regularisation, consistency between observed and permuted statistics |
| Medium | `src/localqtl/stats.py` | The beta approximation and its effect on q-values |

Why now. The localQTL benchmark's calibration gate G2 (job 7693521, 2026-11-08, on this build) failed its
ungrouped-null criterion C1 on one row: `main` `pval_perm` below .001 in 19 of 9,806 null leads against 9.8
expected (one-sided binomial p .0059). The same ledger carries a diagnostic misfit of the dosage scan's
`pval_beta` (Kolmogorov-Smirnov p .0036). Each finding is sorted under the rule set for this review:

1. A reproducible localQTL-specific numerical or statistical defect is fixed before release.
2. A beta-approximation misspecification with acceptable discovery calibration is documented and scheduled.
3. An extreme-tail fluctuation that tensorQTL reproduces and that is not significant after the prespecified
   multiplicity correction is reported and confirmed on an independent null run. It is not a software defect.

## Outcome

| # | Finding | Class | Action |
|---|---|---|---|
| A | Pooled-lead selection: observed scan and permutation null are one procedure | no defect | tests added |
| B | Joint F: formula, degrees of freedom and regularisation correct, identical for observed and permuted | no defect | tests added |
| C | Dosage `pval_beta`: the Beta family misfits the joint-F null slightly, tails conservative | 2 | documented, improvement deferred |
| D | `main` `pval_perm` below .001: 19 against 9.8 | 3 | reported; not reproduced on the independent null run (5 of 9,780 against 9.8); the permutation null matches the exact null |
| E | `calculate_qvalues` set a NaN `pval_beta` to p = 1 instead of using `pval_perm` | 1 (minor, no ledger effect) | fixed |

No kernel and no statistic changed. A scan on the edited source equals the installed 76ba640 build bit for
bit and reproduces the stored benchmark ledger's `pval_perm` and leads exactly (section F).

One condition of class 3 could not be applied as written. The gate carried no prespecified family-wise
correction, so "not significant after the prespecified multiplicity correction" has no rule to refer to
(section D, Multiplicity). Whether C1 passes on this evidence is a ruling for the benchmark, not a result of
this review.

**Addendum, same day: two class 1 defects this review did not find** (section G). An external review of
the branch reported them after the outcome above was written. Both are reproducible and specific to
localQTL, and both are fixed. Neither touches a p-value, a selection rule or a permutation null, so the
conclusions of sections A to D stand, but the sentence "no kernel changed" holds only for the review's own
commits: the fix changes how three kernels compute a reported standard error.

## A. `cis/permutations.py`: observed and permuted pooled-lead selection

`lead_selection="pooled"` picks the lead with the pooled 1-df statistic and tests the dosage model's joint F
there, against a null in which every permutation selects its lead the same way.

- **One design.** `residualize_batch_dosage` puts `y`, the ancestry dosages `D` and the local-ancestry
  columns `L` through one `Residualizer`. The pooled genotype is `G = sum_k D_k` of the residualised
  dosages, which is the residualised genotype by linearity. The observed scan (`_pooled_lead_observed`)
  and the null (`_pooled_lead_null` calling `perm_chunk_r2_argmax`) share one `ctx_p` from
  `prep_ctx_for_perm`: the same Cholesky factor of `L'L` and the same `G_resid2`.
- **Covariate adjustment.** Every permuted chunk is re-residualised (`freedman_lane`, `M P M y`) before the
  selection and before the F. Under `tensorqtl_flavor` the centred raw phenotype is permuted and
  residualised once. The residual degrees of freedom are `n - k_eff - K` for the pooled fit and
  `n - k_eff - (2K - 1)` for the joint fit, with `k_eff = rank(C) + 1`. Both are right.
- **One selection statistic.** The observed path returns the partial R^2 through `t -> t^2 / (t^2 + dof)`,
  the null block computes it directly. On the same vector they agree to 7.5e-9 with the same lead
  (check A1 below).
- **Ties.** `argmax` (observed) and `max(dim=0)` (null) both return the first maximal index. An exact tie in
  pooled R^2 requires the same genotype and the same local ancestry, hence the same dosages and the same
  joint F, so the tie rule cannot move the null statistic. A NaN R^2 is `-1` in the observed scan and `0` in
  the null block. R^2 is non-negative, so neither can win; the difference is cosmetic.
- **One test statistic.** The observed `f_block` and the null F both come from `_dosage_fstat_block` on the
  same `ctx`, with the same sanitisation (NaN to `-1`, an overflowing F kept infinite). `>=` counts a tie
  between observation and null as an exceedance.
- **Grouped scans.** The observed rule (largest pooled R^2 over phenotype and variant, strict `>`, so the
  first phenotype keeps a tie) and the null rule (`r2_sel > r2_sel_global`, the same permutation seed for
  every phenotype of the group) mirror each other.

New tests (`tests/cis/test_permutations.py`):
`test_pooled_lead_null_on_the_identity_permutation_is_the_observed_scan`,
`test_pooled_lead_ties_follow_one_rule_and_cannot_move_the_null_statistic`.

## B. `regression_kernels.py`: the joint F

- `F = b' [A_KK]^-1 b / (K sigma^2)` with `A = (X'X + eps I)^-1`, `X = [D, L]`, referred to
  `F(K, n - k_eff - p)`, `p = 2K - 1`. This is the Wald form of the nested comparison
  `y ~ 1 + C + L` against `y ~ 1 + C + L + D_1..D_K`.
- The block estimator (`_dosage_fstat_block`: the null and `pval_perm`) and the refit
  (`run_batch_dosage_regression`: the reported row) are the same formula. They differ in how the residual
  sum of squares is formed (`y'y - y'XB` against `||y - XB||^2`) and agree to 4.6e-6 relative (check A2).
  The block form loses precision only when the fit is near-perfect, which is why the row reports the refit.
- Regularisation is `1e-6 * mean(diag(X'X))` on `X'X` and `1e-12` relative on `A_KK`. Both live in the `ctx`
  that the observed statistic and every permuted statistic share.
- Independent evidence from the benchmark ledger: the pair-level joint p-value on 25.5 million null pairs
  is .0502 / .0100 / .00102 below .05 / .01 / .001 (KS p .57).

Notes. None is a calibration defect.

- **Empty ancestry stratum.** A variant with no minor allele on one ancestry background has a rank `K - 1`
  design. The ridge keeps the solve finite and the F is referred to `F(K, dof)`, which is conservative for
  that variant. The permutation maximum is unaffected because observed and permuted use the same variants.
- **Two code paths in the correlation scans.** `ancestry_model="main"` and `tensorqtl_flavor` take the
  observed R^2 from the ridge OLS t statistic (`run_batch_regression`) and the null maxima from the
  partial-correlation kernels. On the same vector they agree to 4.7e-5 relative, 5.2e-8 absolute (check A3),
  far inside the resolution of `pval_perm`. The dosage scan already uses one estimator for both.
- **Mixed precision on CUDA.** The default `fp16` autocast covers the null block's matrix products and part
  of the observed `ctx_p` (`H'H`, `H'y`, `H'g`), not `G @ y`. The joint-F block is float32 throughout. The
  benchmark gate ran on CPU, where autocast is off. Before GPU production scans, compare CPU and GPU
  `pval_perm` on one chromosome or pass `mixed_precision="off"`.

New tests: `test_main_model_observed_r2_is_the_null_statistic` (`tests/cis/test_permutations.py`),
`test_dosage_joint_f_block_refit_and_nested_ols_agree_on_a_residualised_design`
(`tests/test_regression_kernels.py`, against the float64 nested-model F with covariates in the design).

## C. `stats.py`: the beta approximation and q-values

The correlation scans follow tensorQTL: an effective degrees of freedom is solved so that the moment
estimate of the first Beta shape is 1, and `Beta(a, b)` is then fitted by maximum likelihood. The dosage
scan (`cis.permutations._beta_fit_on_minp`) fits `Beta(a, b)` directly to the null joint-F p-values at the
nominal `(K, dof)`.

Reproduced on an exact Gaussian null with the package's own kernels
(`docs/reviews/2026-11-09_permutation_inference_checks.py`: 207 samples, 24 covariates, 400 variants in LD,
two ancestries; 40,000 null phenotypes, the Beta fitted on half and scored on the other half):

| Null statistic | Fit | shape1 | KS D (p) | below .05 / .01 / .001 |
|---|---|---|---|---|
| joint lead, maximum F | package (nominal dfd 179) | 0.850 | .0140 (8e-4) | .0437 / .0066 / .0007 |
| joint lead, maximum F | effective dfd 52.9 | 1.066 | .0109 (.017) | .0449 / .0067 / .0006 |
| joint lead, maximum F | empirical p | | .0046 (.79) | .0486 / .0085 / .0012 |
| pooled lead, F at the pooled-selected variant | package (nominal dfd 179) | 0.792 | .0138 (1e-3) | .0431 / .0070 / .0004 |
| pooled lead, F at the pooled-selected variant | effective dfd 38.3 | 1.052 | .0125 (.004) | .0437 / .0066 / .0003 |
| pooled lead, F at the pooled-selected variant | empirical p | | .0045 (.82) | .0470 / .0092 / .0008 |

- The misfit is real, small, and conservative in the lower tail. The benchmark ledger shows the same shape
  on 9,806 null leads: KS D .0179 (p .0036), .0455 / .0082 / .0009 below .05 / .01 / .001, median
  `beta_shape1` 0.78 against 1.05 for `main`.
- The cost grows into the tail. Pooled over the benchmark's two null realisations (S0 and S12, 19,586
  leads), `pval_beta` against `pval_perm` of the same scan:

  | Model | below .05 | below .01 | below .001 |
  |---|---|---|---|
  | `dosage` | 864 / 982 | 132 / 197 | 9 / 20 |
  | `interaction` | 892 / 955 | 151 / 184 | 7 / 17 |
  | `main` | 968 / 1,012 | 162 / 193 | 21 / 24 |
  | tensorQTL | 796 / 847 | 138 / 164 | 16 / 17 |

  Every beta approximation rejects somewhat less often than its permutation p-value (tensorQTL's own about
  16 % less at .01). The dosage scan loses a third at .01 and half at .001. The interaction scan, which
  uses the `true_dof` path and was outside this review, loses as much at .001. This is a cost in power at
  stringent thresholds and no risk to type-I error.
- It is not a consequence of pooled-lead selection. The joint-lead null misfits as much.
- The empirical p (the `pval_perm` construction) is calibrated. `pval_perm` is the reference.
- An effective denominator degrees of freedom, the analogue of tensorQTL's `true_df`, brings `shape1` to 1
  and does not remove the misfit. It is not the improvement to schedule.
- Effect on q-values. `calculate_qvalues` prefers `pval_beta`. On the benchmark the realized FDR at .05 was
  .048 for the dosage scan under the Storey smoother. On the pure null, Storey's pi0 read .935 to .967 under
  the bootstrap and single-lambda estimators, where 1 is the truth: the body misfit reaches pi0 mildly and
  the conservative tail offsets it.

Class 2. Documented in `_beta_fit_on_minp`, `calculate_qvalues`, the user guide and the changelog.

**Finding E.** `calculate_qvalues` replaced a non-finite `pval_beta` by 1.0, which removes the phenotype
from the discoveries whatever its permutation p-value, while `_beta_mle_on_p` documents that callers fall
back to `pval_perm`. Fixed: such a phenotype is tested with its `pval_perm`. The stored columns are not
modified. No benchmark ledger is affected: none of the 143,046 lead records of the null study has a NaN
`pval_beta`.
Tests: `tests/test_stats.py`.

## D. The .001 row of `main` `pval_perm`

Facts from the benchmark's null ledger (S0: `y = C gamma + e`, Gaussian `e`, real genotypes, local ancestry
and covariates; 9,806 leads; 10,000 permutations; `tail_confirmation.md` in the benchmark repository).

| Model | below .05 | below .01 | below .005 | below .002 | below .001 |
|---|---|---|---|---|---|
| expected | 490.3 | 98.1 | 49.0 | 19.6 | 9.8 |
| `main` | 526 (p .053) | 114 (.061) | 65 (.016) | 34 (.002) | 19 (.0059) |
| `dosage` | 498 | 114 (.061) | 67 (.008) | 26 (.096) | 14 (.12) |
| `tq_flavor` | 438 | 98 | 51 | 25 (.14) | 18 (.012) |
| tensorQTL | 440 | 99 | 55 | 28 (.043) | 14 (.12) |

- **Not localQTL-specific.** Of the 19 phenotypes, 10 are below .001 in tensorQTL, 15 below .005 and 18
  below .01, with the same lead variant in 95 %. tensorQTL draws its own permutations under its own scheme.
  A cluster it shares sits in the simulated phenotypes.
- **No code path explains it.** Sections A and B. The simulation adjusts the same covariates it generates
  with, so the null is exact at the pair level, which the nominal joint p-values confirm.
- **Multiplicity.** The gate's prespecified rule is per check (one-sided binomial p below .01 and an excess
  above 20 % of alpha), and the row fails it. The benchmark configuration carried no family-wise
  correction. Over the 12 lead-level tail checks of C1 the row's Bonferroni and Holm p are both .071. This
  is information. It was not the rule when the gate ran.
- **What the ledger cannot separate.** `main` is high at every alpha, and it differs from tensorQTL in both
  the permutation scheme (Freedman-Lane) and the model (a local-ancestry covariate). A chance cluster and a
  permutation null that is slightly light in the tail are both compatible with the table above.

### Confirmation

Two runs in the benchmark repository on this build (76ba640, CPU). The rule and both designs were written
into `configs/calibration.yaml` (`C1_tail_confirmation`) before the replicate was simulated.

**Independent null run** (jobs 7802341 and 7802343, `tail_confirmation.md`). S12 replicates S0: the same
generator, donors, genotypes and covariates, new seeds, 20 cells, 9,780 leads. The prespecified check is
the failing row itself under C1's tail rule (an excess above 20 % of alpha with one-sided binomial p
below .01).

| `main` `pval_perm` | below .05 | below .01 | below .005 | below .002 | below .001 |
|---|---|---|---|---|---|
| S0 (9,806) | 526 | 114 | 65 | 34 | 19 (p .0059) |
| S12 (9,780) | 486 | 79 | 41 | 13 | 5 (p .97) |
| pooled (19,586) | 1,012 | 193 | 106 | 47 | 24 (p .19) |

Verdict: **not reproduced**. On S12 no model meets the tail rule at any alpha. Below .001: tensorQTL 3,
`tq_flavor` 4, `main` 5, `dosage` 6, `interaction` 14 (p .12), each against 9.8. The dosage `pval_beta` has
0 of 9,780 below .001 and 52 below .01 against 97.8, the conservative tail of finding C.

**Scheme diagnostic** (job 7802510, `gaussian_null.md`). For the 114 S0 phenotypes with `main` `pval_perm`
below .01, the same observed statistic was referred to three nulls through `map_permutations`: the ledger's
permutations again, a fresh Freedman-Lane null, and the exact Gaussian Monte Carlo null (vectors `M e*` in
place of permuted residuals), 100,000 draws each. S0 is Gaussian and the scan adjusts the generating
covariates, so the third p-value is exact up to Monte Carlo error.

| Null | below .001 of 9,806 | binomial p |
|---|---|---|
| ledger permutations (10,000) | 19 | .0059 |
| fresh Freedman-Lane (100,000) | 18 | .012 |
| exact Gaussian Monte Carlo (100,000) | 22 | .0005 |

- The rerun equals the ledger's `pval_perm` with largest difference 0, and the three scans name the same
  lead for every phenotype.
- Geometric mean of exact p over Freedman-Lane p: 0.967 (bootstrap 95 % interval 0.940 to 0.996) over the
  114 phenotypes, 0.999 (0.89 to 1.13) over those below .001. The permutation null is not light in the
  tail. Between .001 and .01 its p-values are about 4 % larger than the exact ones (0.962, 0.935 to 0.985).
- The 22 phenotypes are 22 distinct cis windows. The 114 are spread over chromosomes in proportion to the
  phenotypes simulated on each (chi-square p .70).

Reading. The excess does not come from the permutation scheme or from the code that permutes. The S0
phenotypes are this extreme under a null that involves neither: 22 of 9,806 below .001 under the exact
null. That count is the gate's row measured without permutation noise, not a second finding, and taken
alone it is a rarer event than the ledger suggested (p .0005 as a single test, .006 times 12). A defect in
the observed statistic or in the generator would not depend on the seed and would reappear on S12, which
has 5 against 9.8. S0 was simulated on 2026-10-26 and S12 on 2026-11-09 after two changes to the
simulator, so two S0 cells were simulated again with today's code (job 7806511): phenotypes and covariates
are byte-identical to the stored cells. S12 is a replicate of the generator that produced S0.

The evidence supports a fluctuation of the S0 realisation that tensorQTL shares, and it does not support a
software defect. Pooled over both realisations the row is 24 of 19,586 (p .19).

## F. What changed, and what did not

- Tests: five consistency tests (sections A and B) and two q-value tests (section C).
- Fix: the `pval_perm` fallback in `calculate_qvalues` (finding E).
- Documentation: `_beta_fit_on_minp`, `calculate_qvalues`, `docs/localqtl.rst`, `CHANGELOG.md`.
- Not changed: every kernel and every statistic. Checked on the first 40 phenotypes of benchmark null
  task 1 (S0, 10,000 permutations, CPU) for `tq_flavor`, `main` and `dosage`:
  - edited source against the installed 76ba640 build on one node (job 7803585): every numeric output
    column identical bit for bit (18, 18 and 43 columns), same leads, same NaN pattern;
  - edited source against the stored ledger (job 7803526): `pval_perm` and leads identical, `pval_beta`
    within 4e-7, 7e-7 and 6e-6. The installed build shows the same three differences, so they come from
    rerunning the beta fit on another node, not from the edits.

Deferred to a later version, each needing its own evaluation:

- a null fit for the F-based scans that is accurate where discoveries are made (a tail-focused fit or a
  different family), validated against the exact Gaussian null used here;
- one estimator for the observed and the null statistic in the correlation scans, as the dosage scan has;
- a rank-aware reference distribution for variants with an empty ancestry stratum.

## G. Addendum: two defects found after the review

Reported by an external review of `feat/pooled-lead-selection` on 2026-11-09 and confirmed here.

**G1. Standard error of a negative effect (class 1).** `prep_ctx_for_perm` and both branches of
`run_batch_regression_with_permutations` computed `se = |beta| / clamp(t, 1e-8)`. For a negative t the
clamp returns 1e-8 and the standard error becomes `|beta| * 1e8`.

- Reach, measured by running each scan with a negative effect on the 76ba640 build: `slope_se_pooled`
  (new in 0.4.4), and in released versions the `slope_se` of grouped `map_permutations` scans, of the
  g x local-ancestry and g x covariate interaction scans, and of the signals `map_independent` adds. In
  the benchmark's stored null runs the `interaction` model's lead `slope_se` has a median of 9e7 where
  the slope is negative and 0.29 where it is positive.
- Not affected: slopes, t, R^2, every p-value, the choice of lead; the ungrouped `main` scan, the nominal
  scans and the dosage model (other kernels). No benchmark ledger reads the affected columns.
- Why the review missed it: section A checked the selection statistic (R^2 from t^2) and the p-values of
  `_pooled_lead_observed`, not the standard error it passes through, and the existing tests compared
  `slope_pooled` and `tstat_pooled` with the main model on a positive effect only.
- Fix: one helper, `_slope_se`, computes `sqrt(RSS / dof / SS_g)` from the residual sums of squares in all
  three places. Tests cover both signs at the kernel level and through the pooled, interaction, grouped
  and independent scans.

**G2. LD-aware refit with no residual degrees of freedom (class 1).** `ld_aware_contrast` projects a
conditioning block of unrestricted rank out of the lead's design and calls
`run_batch_dosage_regression` with `k_eff + rank`. The kernel floors `n - k_eff - p` at 1, so a block
that exhausts the sample still returned statistics. Reproduced on a null phenotype (n 24, 18 residual
dimensions): F 1.5e8, joint p 6e-5, contrast p 8e-5 at `dof_ld` 1.

- Fix: when `n - k_eff - rank - p < 1` the function returns every inferential column as NaN with
  `dof_ld = 0`, and keeps `n_conditioned_ld` and `rank_ld`. With one residual degree of freedom left it
  still reports, and matches numpy's count.
- Not changed: the kernels' own floor of 1, which the unconditioned scans share. There it can only
  engage when the covariates alone exhaust the sample.

Checked after both fixes: the full suite passes (247 passed, 5 skipped). Eight permutation scans (`main`
with and without a panel, both interaction models, the pooled-lead dosage scan with and without
`ld_aware`, grouped `main` and grouped interaction; 12 phenotypes with effects of both signs) were run on
the 76ba640 build and on the fixed source: every column other than the standard errors named above is
identical bit for bit, the standard errors of negative effects go from about 1e7 to their OLS values, and
those of positive effects differ by at most 1.2e-7 relative.

## Reproduce

```
# numeric checks A1-A3 and the table of section C (CPU, seconds)
python docs/reviews/2026-11-09_permutation_inference_checks.py

# the tests that pin the conclusions
pytest tests/cis/test_permutations.py tests/test_regression_kernels.py tests/test_stats.py

# section D, in the localQTL-benchmarking repository
bash 04_calibration_power/_h/run_stage.sh --only 03c,04b
```
