# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions follow the git tags (`vX.Y.Z`); the package version is derived from the tag.

To release: rename the `## [Unreleased]` heading to `## [X.Y.Z] - YYYY-MM-DD`, update the
compare links at the bottom, merge, and only then tag `vX.Y.Z`. The release workflow builds
the GitHub release body from that section and stops if the tag has none.

## [Unreleased]

### Changed

- **The ancestry-specific dosage scan has two explicitly distinguished inference products, and
  neither rests on the Beta approximation.** (1) Phenotype-level discovery: the calibrated
  empirical permutation p-value `pval_perm` of the selected-lead null is the phenotype-level
  p-value (`pval_fit_method = "empirical"`, the dosage scan's default); no Beta is fitted and
  `pval_beta`, `beta_shape1`, `beta_shape2` are NaN. `calculate_qvalues` computes the q-values on
  it (the path it already took for a NaN `pval_beta`). (2) Variant-level extraction of joint
  associations: the scan stores a compact summary of each phenotype's permutation null, the
  smallest null p-values exactly and quantiles on a geometric grid, of both the selected-lead
  null and the window-maximum joint-F null (`result.attrs["null_summary"]`, read with
  `localqtl.stats.null_summary_of`; the CLI writes `<prefix>.cis_qtl.null_summary.parquet`), and
  `calculate_qvalues` sets `pval_nominal_threshold` to the gene-level cutoff's quantile of the
  window-minimum null (`localqtl.stats.empirical_nominal_threshold`: the k-th smallest null
  value, k = floor(p* (nperm + 1)), exact within the stored tail and log-interpolated above it;
  NaN when p* is below the permutations' resolution). `get_significant_pairs` gains
  `pair_pval_col`, the per-pair statistic the threshold applies to; the CLI's `cis_nominal
  --cis_output` and `CisMapper.get_significant_pairs` pick `pval_joint` for a dosage table
  (`localqtl.cis.postproc.pair_pval_column_of`), so the lead-level test, its null and the pair
  statistic are one joint-F procedure. Before, the pairs of a dosage eGene were selected by
  comparing the nominal table's pooled 1-df `pval_nominal` with a threshold derived from the Beta
  fit of the joint-F null: two different statistics. With `lead_selection="pooled"` the two
  nulls differ (the F at the permutation's pooled lead against the maximum over the window) and
  are read from the same F block (`regression_kernels.dosage_perm_chunk_f_at_and_max`), so the
  pooled-lead heterogeneity procedure and the joint-F model are unchanged. Motivation (localQTL
  benchmark, 2026-11-09, `04_calibration_power/05_metrics/_m/discovery_source.*`): the Beta of
  this F-based null is conservative in the tail (0.4.4 Known limitations); on the power
  scenarios q-values on `pval_perm` found 11.6 % more dosage discoveries at FDR .05 (20,652 to
  23,043 over 1,340 cells) with the realized false-discovery rate at the other models' level,
  and nothing changed on the null scenarios. `pval_fit_method="beta"` keeps the legacy Beta
  product for the dosage scan, with its threshold from the Beta quantile. The correlation scans
  (`main`, `interaction`, `tensorqtl_flavor`) keep `"beta"` as their default and are unchanged;
  `beta_approx=False` is `"empirical"` for every scan. Every permutation table now carries
  `pval_fit_method` per row (the CLI: `--pval_fit_method {beta,empirical}`); tables written
  before 0.4.5 are read as `"beta"` where `pval_beta` is finite.
- `get_significant_pairs` selects phenotypes with `qval <= fdr` and keeps pairs with the pair
  statistic `<= pval_nominal_threshold` (both boundaries included, as `calculate_qvalues` counts
  its discoveries); before, both comparisons were strict.
- `calculate_qvalues` derives the gene-level cutoff p* from the phenotype-level p-value of every
  row (so a phenotype tested on `pval_perm` because its Beta fit failed takes part in it); for a
  table whose rows all carry Beta parameters nothing changes.

### Fixed

- `calculate_qvalues(qvalue_lambda=x)` with a single lambda (the CLI's `--qvalue_lambda`) raised
  inside `py_qvalue`, whose pi0 estimator takes a grid; a scalar lambda is now Storey's pi0 at
  that lambda (R's `qvalue(lambda = x)`), passed to the q-value computation directly, and a pi0
  of 0 is reported as such. The local FDR, which no caller used, is no longer requested: its
  kernel density failed on tables with tied or very few p-values, which the empirical p-value of
  a small scan produces.

### Under evaluation (not in this release)

- A generalized-Pareto extrapolation of the permutation tail (Knijnenburg et al. 2009) below the
  resolution `1 / (nperm + 1)` of the empirical p-value. It becomes the default only if it passes
  the localQTL benchmark's pre-registered validation (`configs/calibration.yaml:
  discovery_source.gpd_acceptance`: an independent null, both lead rules, 10^6-permutation
  references, the empirical/GPD transition, mixed-null FDR, and a material gain in discoveries).

## [0.4.4] - 2026-11-09

### Added

- **Pooled-lead selection for the dosage scan** (`map_permutations(ancestry_model="dosage",
  lead_selection="pooled")`, CLI `--lead_selection pooled`). The lead of each phenotype is the
  variant the pooled 1-df statistic of `y ~ G + L` picks on the dosage model's own residualised
  design, i.e. the lead of `ancestry_model="main"`, and the row reports the dosage model's own
  joint test at that variant: `pval_nominal` is still `pval_joint`, and `pval_perm` / `pval_beta`
  still calibrate the joint F, but against a null in which every permutation selects its lead by
  the pooled statistic too and the joint F is read at that variant
  (`regression_kernels.perm_chunk_r2_argmax`, `dosage_perm_chunk_f_at`), so the p-value is
  calibrated for the selection procedure; the maximum F over the window (the joint-lead null)
  would be conservative at a variant the joint F did not pick. The pooled model's p-value is never
  substituted for the dosage model's; it is reported as information in `pval_nominal_pooled`
  with `slope_pooled`, `slope_se_pooled`, `tstat_pooled` and `r2_pooled`. `ld_aware` evaluates
  the LD-aware contrast at the pooled lead; with `group_s` the group's lead is the (phenotype,
  variant) with the largest pooled R^2 and the null selects the same way. The default,
  `lead_selection="joint"`, is the 0.4.3 scan, unchanged. Motivation (localQTL benchmark,
  2026-11-06): on the shared-effect null the ancestry contrast at the joint lead read 42 % below
  .05 against 4.3 % at the pooled lead, because the joint F prefers variants whose marginal
  ancestry effects differ. Tests: the pooled lead is the main model's lead and the row's test the
  nominal dosage scan's joint test there; the kernel-level null is uniform where the joint-lead
  null is conservative; calibration under the null, ungrouped and grouped; the CLI and the
  mapper.

### Fixed

- **Standard errors of negative effects from the permutation kernels.**
  `prep_ctx_for_perm` and `run_batch_regression_with_permutations` derived the slope's standard
  error as `|beta| / clamp(t, 1e-8)`, which is `|beta| * 1e8` whenever the effect is negative. The
  reported slope, standard error and t statistic were then inconsistent in every table that takes
  its lead statistics from these kernels: `slope_se_pooled` of the pooled-lead dosage scan (new in
  this release) and, in earlier releases too, `slope_se` of grouped `map_permutations` scans, of
  the g x local-ancestry and g x covariate interaction scans (with `slope_se_gxh_anc{k}`), and of
  the signals `map_independent` adds. Slopes, t statistics, R^2, every p-value and the choice of
  lead were not affected, nor were the ungrouped `main` scan, the nominal scans and the dosage
  model's own statistics. The standard error is now computed from the residual sums of squares
  (`sqrt(RSS / dof / SS_g)`), so it no longer depends on the sign of the effect and is positive at
  a zero effect; standard errors of positive effects change only in the last float digits. Tables
  written by an earlier version need their `slope_se` recomputed as `|slope / tstat|` wherever the
  slope is negative.
- **LD-aware refits with no residual degrees of freedom are not reported.** A credible set broad
  enough for its conditioning block to exhaust the sample (`n - k_eff - rank_ld - p < 1`, with `p`
  the lead's own design columns) left a saturated conditional model whose residual is numerical
  noise, and the regression kernel's dof floor of 1 turned that into finite, sometimes extreme
  statistics (joint F near 1e8 and p near 1e-5 on a null phenotype). `ld_aware_contrast` now
  returns every `*_ld` slope, standard error, t, F, z and p-value as NaN with `dof_ld = 0` in that
  case; `n_conditioned_ld` and `rank_ld` still describe the block. Both entry points
  (`map_permutations(ld_aware=True)`, `map_conditional_dosage(credible_sets_df=...)`) go through
  it; the marginal columns of the row are unchanged.
- `CisMapper.map_permutations` now accepts `ld_aware`, `ld_aware_opts` (documented for 0.4.3 but
  only reachable through the function API) and `lead_selection`.
- `calculate_qvalues` tests a phenotype whose beta approximation could not be fitted (`pval_beta`
  NaN) with its `pval_perm`, as `_beta_mle_on_p` documents, instead of setting it to p = 1, which
  removed it from the discoveries whatever its permutation p-value. The stored p-value columns are
  not modified and the phenotype keeps a NaN `pval_nominal_threshold`; with no `pval_perm` column
  the p = 1 fill is unchanged. Tables without NaN `pval_beta` are unaffected.

### Known limitations

- **Beta approximation of the dosage scan.** `pval_beta` of `ancestry_model="dosage"` is a
  two-parameter Beta fitted by maximum likelihood to the null joint-F p-values at the nominal
  `(K, dof)` (`cis.permutations._beta_fit_on_minp`), without the effective-degrees-of-freedom step
  of the correlation scans. The Beta family does not reproduce this null exactly: `beta_shape1`
  sits near 0.8 rather than 1 (median 0.78 against 1.05 for `ancestry_model="main"` on the
  localQTL benchmark's null), the body of the null distribution is slightly misfitted
  (Kolmogorov-Smirnov distance 0.012-0.018 on 10,000-20,000 null phenotypes) and the lower tail is
  conservative (0.042-0.046 below .05, 0.005-0.008 below .01). The cost grows into the tail: on
  the benchmark's 19,586 null leads `pval_beta` falls below .01 for 132 phenotypes against 197
  for `pval_perm`, and below .001 for 9 against 20 (`ancestry_model="main"`: 162 against 193 and
  21 against 24). It is not a consequence of
  `lead_selection="pooled"`: the joint-lead null misfits as much, on an exact Gaussian null drawn
  with the package's own kernels. `pval_perm` is calibrated and is the reference; q-values on
  `pval_beta` held the nominal FDR on the benchmark (realized .048 at .05). Solving for an
  effective denominator dof that sets the moment `shape1` to 1, the analogue of tensorQTL's
  `true_df`, was tried and does not remove the misfit, so a replacement null fit is deferred to a
  later version. Documented in the user guide and in `_beta_fit_on_minp`.

### Review

- **Permutation inference reviewed (2026-11-09,
  `docs/reviews/2026-11-09_permutation_inference.md`).** The observed scan and its permutation
  null are one procedure in the pooled-lead dosage scan (one residualised design, one selection
  statistic, the first-index tie rule on both sides, re-residualised permuted phenotypes, one
  joint-F estimator with the same sanitisation); the joint F, its degrees of freedom and its
  regularisation are those of the nested OLS comparison and are identical for observed and
  permuted statistics. No kernel or statistic changed. New tests pin the conclusions: the
  pooled-lead null on the identity permutation returns the observed scan's numbers; an exact
  proxy ties in pooled R^2 and carries the same joint F under every permutation; the `main` /
  `tensorqtl_flavor` observed R^2 (ridge OLS) equals the null kernels' statistic; the block and
  refit joint F equal the float64 nested-model F on a design with projected-out covariates.

## [0.4.3] - 2026-11-04

### Added

- **LD-aware ancestry contrast** (`localqtl.cis.ld_aware`). The pairwise contrast of the dosage
  model, `pval_diff_anc{i}_anc{j}`, compares *marginal* ancestry-specific effects; at a lead that
  tags the causal variant it fires whenever the LD between the two differs by ancestry, even when
  the causal effect is the same in every ancestry (on the localQTL benchmark's shared-effect
  design, 42 % of the joint leads of the permutation scan had `pval_diff` below .05 against 5.3 %
  at the causal variant). `map_permutations(ancestry_model="dosage", ld_aware=True,
  ld_aware_opts=...)` fits SuSiE to the pooled genotypes of each window and reports the lead's
  contrast conditional on the other members of the phenotype's credible sets (their pooled
  genotypes, `condition_on="genotype"`, or their ancestry-specific dosages, `"dosage"`) as
  `slope_ld_anc{k}`, `slope_se_ld_anc{k}`, `tstat_ld_anc{k}`, `pval_ld_anc{k}`, `fstat_joint_ld`,
  `pval_joint_ld`, `z_diff_ld_anc{i}_anc{j}`, `pval_diff_ld_anc{i}_anc{j}`, `n_conditioned_ld`,
  `rank_ld`, `dof_ld`, `n_cs_ld`, `cs_size_ld` and `pip_ld`, next to the unchanged marginal
  columns; `pval_perm` and `pval_beta` are unaffected. `map_conditional_dosage(...,
  credible_sets_df=...)` adds the same columns at known signals, conditioning on the other
  signals and on the credible-set variants (`n_conditioned_cs`). The conditioning block is
  orthogonalised against the lead's own genotype and reduced to its rank, so a conditioning
  variant in perfect LD with the lead costs no degree of freedom and leaves the lead
  identifiable; the contrast, its standard error and `dof_ld` are those of the full conditional
  regression. CLI: `--ld_aware` (`--mode cis` with `--ancestry_model dosage`; `L` from
  `--max_effects`, the mode from `--condition_on`) and `--credible_sets <table>` (`--mode
  cis_conditional`; the `cis_susie` summary is accepted as is). `tests/cis/test_ld_aware.py`
  checks the contrast against a full conditional OLS in both modes, the rank reduction, the
  tagging leak against the window's credible sets, and both entry points.

### Fixed

- **SuSiE credible sets on the GPU.** `susie_get_cs` holds the member indices as tensors on the
  compute device, subsampled a set larger than 100 members with `np.random.choice` before the
  purity filter and returned the pure sets with `.numpy()`; both raise "can't convert cuda:0
  device type tensor to numpy" on a GPU (`cis_susie`, and the LD-aware contrast on its first GPU
  run). The subsample is now drawn on the host and indexed on the device, and the returned sets
  are copied to the host; the CPU path and its random stream are unchanged.

## [0.4.2] - 2026-10-26

### Fixed

- **`tensorqtl_flavor` now uses tensorQTL's permutation scheme.** 0.4.1 re-residualised the
  permuted covariate residual in every scan (Freedman & Lane, `M P M y`), and said that this
  restored parity with tensorQTL. It does not: tensorQTL permutes the *raw* phenotype and
  residualises the permuted copies once (`M P y`; `calculate_cis_permutations` permutes
  `phenotype_t`, `calculate_corr` applies the residualizer). Both schemes are valid under
  exchangeability but they are different nulls: on 207 caudate samples with 24 covariates and
  10,000 permutations, `M P M y` gives 3-7% more eGenes per chromosome than tensorQTL on
  identical input (beta shape2 about 3% higher, `pval_beta` Spearman 0.995 against tensorQTL),
  and tensorQTL fed pre-residualised phenotypes reproduces those numbers exactly, so the gap is
  the scheme and nothing else. Freedman-Lane stays the default for every model.
  `tensorqtl_flavor=True` (`map_permutations`, `map_independent`, grouped and ungrouped,
  with paired covariates and conditioning signals) now permutes the centred raw phenotype and
  residualises it once with the bases of every residualizer in scope
  (`localqtl.cis._permute.perm_source`, the `y_source` argument of `compute_perm_r2_max`),
  which is tensorQTL's null up to the random streams (`tests/cis/test_permutation_scheme.py`
  checks both schemes against a numpy reference on the same permutation indices and the flavor
  against `tensorqtl.core.calculate_corr`). Nominal statistics are unchanged by the scheme.
- **`true_dof` no longer overflows.** The effective-degrees-of-freedom root finder could run
  off (1e14 on an interaction permutation scan of a null phenotype) and the value was stored in
  an `int32` column, which raised `OverflowError` mid-scan and lost the chromosome. `true_dof`
  is now `float32` in every result table, as tensorQTL's `true_df`, and an estimate more than a
  factor `localqtl.stats.TRUE_DOF_MAX_RATIO` (10) away from the nominal degrees of freedom falls
  back to the nominal value, as an optimiser failure already did.

## [0.4.1] - 2026-10-25

### Fixed

- **Permutation p-values were anti-conservative in proportion to the number of covariates.**
  Every permutation scan (`map_permutations` with and without local ancestry, the dosage
  joint-F scan, grouped scans and `map_independent`) permuted the covariate-residualised
  phenotype and compared it with the residualised genotypes without projecting the permuted
  values back onto the covariate-orthogonal subspace. A permuted residual vector has `n`
  exchangeable entries, the observed statistic lives on `n - k - 1` residual degrees of
  freedom, so the null maximum statistic was too light (expected squared correlation
  `1/(n-1)` instead of `1/(n-k-2)`), and `pval_perm` / `pval_beta` were too small: with
  207 samples and 24 covariates the fraction of null phenotypes below 0.05 was 0.13-0.15
  (0.19-0.27 for grouped real transcripts), and the scan called about twice tensorQTL's
  eGenes on the same input. Permuted phenotypes are now re-residualised (Freedman & Lane,
  1983: `M P M y`, `localqtl.cis._permute.freedman_lane`) with the covariate bases of every
  residualiser in scope (shared covariates, paired covariates, the conditioning signals of
  `map_independent`). tensorQTL permutes the raw phenotype and re-residualises, so parity
  with it is restored up to the random streams. Nominal statistics are unchanged.

## [0.4.0] - 2026-10-25

### Added

- **Permutations for the ancestry-specific dosage model.**
  `map_permutations(ancestry_model="dosage")` calibrates the joint signal: the per-variant
  statistic is the joint Wald F over the K ancestry effects and the permutation statistic is
  its maximum over the cis-window, so `pval_perm` and `pval_beta` apply to that joint test
  while the per-ancestry columns of `map_nominal` (`slope_anc{k}`, `pval_joint`,
  `z_diff_anc{i}_anc{j}`, `an/ac/af/ma_count_anc{k}`) are reported for the lead variant.
  Because the test is an F test rather than a correlation, the Beta approximation is fitted
  to the permuted minimum p-values directly, so `true_dof` is the residual degrees of freedom
  and `pval_true_dof` is `NaN`. Null phenotypes give uniform `pval_perm` and `pval_beta`
  (verified in the test suite). `min_ancestry_ac` masks the reported per-ancestry estimates
  without changing the joint test or the variant ranking. `interaction_covariate` and
  `paired_covariate_df` are rejected in this mode, and `map_independent` still does not
  implement it.
- **Phenotype groups in the dosage permutation scan.** `map_permutations(ancestry_model="dosage",
  group_s=...)` takes the maximum joint F over the phenotypes of a group as well as over the
  window and returns one calibrated row per group, which is the layout wanted for isoform and
  splicing-event phenotypes grouped by gene. The reduction is valid because the phenotypes of
  a group share samples, covariates, window and K, so their residual degrees of freedom are
  identical and the largest F is the smallest joint p-value. The per-variant inverse
  cross-products are built once per group and reused across its phenotypes and every
  permutation chunk, so a grouped scan costs less per phenotype than an ungrouped one, and
  permutations are drawn once per group so the within-group correlation is preserved. As in
  the correlation-based grouped scan it mirrors, the null permutes covariate-residualized
  phenotypes, and a maximum over the group amplifies that approximation: in a null study
  (50 designs x 60 groups, 80 samples, 12 variants, 500 permutations) the fraction of
  `pval_perm` below 0.05 rose from 0.059 at group size 1 to 0.070 at size 3 and 0.073 at
  size 5, against 0.056 / 0.062 / 0.064 for `ancestry_model="main"`. Grouped p-values from
  large groups are mildly anti-conservative in both models; this is documented, not fixed.
- **`localqtl.cis.map_conditional_dosage`: ancestry-specific effects at known signals.**
  Takes signals that were already selected (the output of `map_independent`, or the lead rows
  of `map_permutations`) and re-fits each with the dosage model, conditioning on the other
  signals of the same phenotype. It reads only the listed variants and needs no permutations,
  because selection has already happened. `condition_on` chooses what to project out for the
  other signals: `"genotype"` (default) their pooled dosage, matching how `map_independent`
  defined them and costing one degree of freedom each; `"dosage"` their K ancestry-specific
  dosages, so an ancestry-differential secondary signal cannot leak into the variant under
  test, at K degrees of freedom each; `"none"` fits every signal marginally. All three modes
  reproduce the corresponding numpy OLS fit, degrees of freedom included. The output adds
  `n_conditioned` and `dof` to the per-ancestry columns; its p-values are conditional nominal
  p-values at variants chosen on the same data, so they measure how different the
  ancestry-specific effects of an established signal are, not whether the signal exists.
  Exposed as `CisMapper.map_conditional_dosage` and the `cis_conditional` CLI mode
  (`--condition_on`, writing `<prefix>.cis_conditional_qtl.txt.gz`).
- New kernels `prep_ctx_for_dosage_perm`, `dosage_fstat` and `dosage_perm_chunk_fmax`: the
  per-variant `(X'X)^-1` and the inverse of its ancestry block do not depend on the
  phenotype, so each permutation chunk costs a few batched matmuls. They reproduce
  `run_batch_dosage_regression` to float32 precision.
- **Phenotype-specific covariates in `map_independent` and `susie.map`** (`paired_covariate_df`,
  `--paired_covariate` on the command line, also for the `cis_independent` and `cis_susie`
  modes), matching the support added to `map_nominal` and `map_permutations` in 0.2.0.
  Phenotypes absent from the table use the shared covariates alone; phenotype groups are not
  supported. A paired covariate is exactly equivalent to adding that column to `covariates_df`
  for that phenotype.
- `CisMapper.map_permutations` forwards `paired_covariate_df` and `min_ancestry_ac`, and
  `CisMapper.map_independent` now forwards `paired_covariate_df` too.
- The CLI accepts `--ancestry_model dosage` in `cis` mode as well as `cis_nominal`, and passes
  `--min_ancestry_ac` to the permutation scan.

### Changed

- The vectorised dosage column computation moved out of `_run_nominal_core` into
  `localqtl.cis.nominal.dosage_result_arrays`, which `map_nominal`, `map_permutations` and
  `map_conditional_dosage` now share. No change to any reported value.
- `susie.map` takes `paired_covariate_df` as a keyword-only argument at the end of the
  signature instead of the sixth positional one, so that `L` keeps its position. Callers that
  passed `L` positionally would otherwise have bound it to `paired_covariate_df` and crashed.
  The parameter is new in this release, so no code written against 0.3.0 or earlier is
  affected.
- `auto_perm_chunk` moved from `localqtl.cis.independent` to `localqtl.cis.common`, shared
  with the dosage permutation paths.

### Fixed

- **`map_conditional_dosage` silently dropped signals and under-conditioned the rest.** The
  variant sub-frames were ordered by variant ID, so the position column the cis-window
  generator bisects was not monotonic whenever lexicographic order disagreed with position
  order (`chr1:1000000:A:G` before `chr1:999999:A:G`, `v10` before `v2`, most rsID sets). The
  affected signals were reported as "monomorphic or unmapped" and dropped, and the signals
  that survived were then fitted with fewer conditioning variants than requested --
  `n_conditioned=0`, i.e. marginally, in the two-signal case. The sub-frames are now ordered
  by `(chrom, pos)`.
- **`map_conditional_dosage` measured `start_distance`/`end_distance` from the wrong origin.**
  `phenotype_pos_df` was accepted and never used; the distances were measured from the span
  of a phenotype's signals rather than from the phenotype itself, so they did not agree with
  the identically named columns of `cis`, `cis_nominal` and `cis_independent` output. They
  now come from `phenotype_pos_df`.
- `map_conditional_dosage` returns the full column schema when no signal survives filtering,
  including `rank` and the per-ancestry dosage columns, instead of 13 fixed columns that made
  `out["pval_joint"]` raise `KeyError` on an empty result.
- **The dosage permutation null discarded its most extreme draws.** `dosage_perm_chunk_fmax`
  mapped an overflowing (`+inf`) F to `0.0` while the observed statistic kept it, so a null
  permutation more extreme than the observation was counted as the least extreme one and
  `pval_perm` came out anti-conservative. Null and observed statistics now use the same
  sanitisation (NaN never wins the maximum; an overflow does).
- **`pval_nominal` and `pval_joint` of a dosage permutation row no longer disagree.** The
  first came from the batched normal-equations F, which forms `y'y - y'XB` in float32 and
  cancels catastrophically once the fit is near-perfect, while the second came from the
  explicit-residual fit. Both now come from the stable fit; the permutation p-value is still
  compared against the null on the null's own estimator, as it must be.
- The `cis_conditional` CLI mode rejects `--chunk_size` (it previously crashed with
  `AttributeError: 'NoneType' object has no attribute 'index'`) and `--phenotype_groups`
  (previously read and then silently ignored).
- `susie.map` builds a phenotype's `Residualizer` when it reaches that phenotype instead of
  building one per row of `paired_covariate_df` up front, which held a QR factor per
  phenotype on the device for the whole run (~1.2 GB for 20,000 phenotypes x 500 samples x 30
  covariates).
- The dosage permutation paths refit only the lead variant rather than the whole cis-window,
  and size their permutation chunks with `auto_perm_chunk` as the correlation-based paths do;
  the per-chunk `(m, p, C)` buffers previously scaled with the raw `perm_chunk`.
- `map_conditional_dosage` reuses one `Residualizer` for every signal whose design is just
  the covariate matrix (`condition_on="none"`, or a phenotype with a single signal).
- `validate_ancestry_model` no longer reports that `ancestry_model='dosage'` is "only
  implemented for map_nominal"; it names the caller and the mappers that do support it.

## [0.3.0] - 2026-10-24

External validation of lead variants against the AlphaGenome Atlas and the IGVF
Catalog, and a complete autodoc API reference.

### Added

- **`localqtl.annotate.alphagenome`: AlphaGenome Atlas molecular-effect scores.**
  `score_lead_variants` queries the Atlas for the lead variants (live client via the
  `alphagenome` SDK and an API key in `$ALPHAGENOME_API_KEY`, batched, with an optional
  `cache_path`), `read_atlas_scores` / `write_atlas_scores` handle exported tidy-score tables
  so an analysis reruns offline, and `annotate_with_alphagenome` joins the scores onto results
  by `(chrom, pos, ref, alt)` and gene (version suffixes ignored), adding `ag_raw_score`,
  `ag_quantile_score`, `ag_biosample`, `ag_n_tracks` and `ag_sign_concordant`. The SDK is
  licensed for non-commercial use and is not locked as a dependency; install it separately.
- **`localqtl.annotate.igvf`: IGVF Catalog regulatory links.**
  `query_variant_gene_links` and `query_element_gene_links` page through the public REST API
  (region queries of at most 10 kb, retries on 429/5xx, optional method / source / biosample
  filters and a `cache_path`), `read_bedpe_links` reads downloaded ENCODE-rE2G / scE2G
  prediction sets, and `read_links` / `write_links` round-trip cached tables.
  `compare_with_regulatory_links` flags QTL pairs whose variant has a direct variant->gene link
  or lies in an element linked to the same gene (`n_links`, `same_gene_support`, `best_method`,
  `best_source`, `distance_to_element`, and `supported_anc{k}` for the dosage model), and
  `summarize_link_concordance` reports the supported fraction per ancestry, method or source.
  The Catalog reports 0-based positions; every position returned by the client is 1-based, as
  in `variant_df`.
- Optional extra `annotate-live` (`requests`) for the IGVF REST client; the cached-table and
  BEDPE readers need nothing beyond the core dependencies.
- `docs/api.rst`: autodoc pages for every public module, linked from the documentation index.
- A `live` pytest marker for the tests that call external APIs; they are skipped unless
  `pytest --run-live` is given.

### Changed

- Documentation: the Sphinx build no longer replaces heavy dependencies with empty module
  stubs (they broke `import localqtl.cis.nominal` during autodoc, so no API member was
  documented); unavailable GPU-only and optional packages are mocked by `autodoc_mock_imports`
  instead, and `nitpick_ignore_regex` silences the free-text parameter types numpydoc renders
  as cross-references. Several docstrings that numpydoc mis-parsed (a `Returns` description on
  the type line) were rewritten.

## [0.2.0] - 2026-10-24

Ancestry-resolved cis-xQTL mapping with external allele-frequency QC, and
feature parity with tensorQTL (trans mapping, eigenMT, multi-term interactions,
post-processing and a command line). Two regression fixes change numeric
output; see *Upgrading from 0.1* in the README.

### Added

- **Posterior-aware ancestry-specific dosage model.** `map_nominal(ancestry_model="dosage")`
  fits one effect per ancestry from the expected ancestry-specific alternate-allele dosage
  (`D_k = G * L_k / 2`), reports `slope_anc{k}`, the joint test `fstat_joint` / `pval_joint`,
  pairwise `z_diff_anc{i}_anc{j}` / `pval_diff_anc{i}_anc{j}` from the full covariance of the
  ancestry betas, and the per-ancestry in-sample allele statistics `an_anc{k}`, `ac_anc{k}`,
  `af_anc{k}`, `ma_count_anc{k}`. `min_ancestry_ac` masks ancestries with too few minor alleles.
  `LocalAncestryPanel(use_posterior=True)` serves posterior-expected counts from RFMix `.fb` output.
- **Allele metadata.** `PgenReader.variant_df`, `PlinkReader.variant_df` and `load_genotypes`
  carry `chrom, pos, ref, alt` with `alt` the counted (dosage) allele;
  `genotypeio.alleles_from_variant_ids` parses `chr1_123_A_G_b38`-style IDs.
- **`localqtl.freq`: external allele frequencies and ancestry-aware QC.** `FrequencyTable`
  (GRCh38, keyed by `chrom, pos, ref, alt`, allele-aware lookup that re-orients swapped alleles
  and flags strand-ambiguous sites), importers for gnomAD v4 sites VCFs (including the
  local-ancestry-tract fields, detected from the header), the gnomAD local-ancestry release,
  NCBI ALFA, and `bcftools query` output (no pysam needed); `ancestry_allele_frequencies`
  (in-sample per-ancestry AF), `compare_allele_frequencies` / `summarize_af_concordance`,
  `annotate_with_frequency`, frequency- and ancestry-stratified calibration
  (`calibration_by_frequency`, `calibration_by_frequency_dosage`, genomic-control lambda,
  QQ tables) and `run_frequency_benchmark` with a permutation null. Ancestry labels map to
  reference groups only through an explicit `ancestry_map`. Optional extra `annotate` (pysam)
  for tabix access.
- **`map_nominal_loci`** (exported; pads the window exactly once; several loci per phenotype)
  and `CisMapper.map_nominal_loci`.
- **tensorQTL parity.**
  - `localqtl.trans`: `GenotypeGeneratorTrans`, `map_trans` (sparse or dense output, single
    `interaction_s`, optional `LocalAncestryPanel` main-effect covariates), `filter_cis`,
    `map_trans_permutations` (genome-wide or leave-one-chromosome-out null) and
    `apply_permutations`.
  - `localqtl.eigenmt`: `lw_shrink`, `compute_tests`, `run_eigenmt`, `padjust_bh`.
  - `map_nominal`: `interaction_df` (one or several terms, tensorQTL column naming),
    `maf_threshold_interaction`, `paired_covariate_df`, `write_top` / `write_stats`
    (`--best_only`), `run_eigenmt` and `top_assoc_path`; the top association per phenotype
    (or group) is written to `<out_prefix>.top_assoc.txt.gz` or returned in
    `results.attrs["top_assoc"]`. `map_permutations` accepts `paired_covariate_df`.
  - `localqtl.post`: `calculate_afc` (mixQTL total-read-count model), `calculate_replication`
    (with `compute_pi1`, interaction and paired covariates), `annotate_genes`, `linreg`, `trc`;
    re-exports `calculate_qvalues` and `get_significant_pairs`.
  - `localqtl.compat`: `to_tensorqtl_columns` / `from_tensorqtl_columns`
    (`true_dof` <-> `true_df`, `se_gi` <-> `b_gi_se`, ...).
  - `genotypeio.get_sample_ids`, `parse_genotypes`, `get_vcf_region`, `get_vcf_variants`
    (pysam when installed, otherwise the `tabix` binary); `pgen.get_chunk_size` and
    `pgen.generate_paired_chunks`.
  - `stats.fit_beta_parameters`; `preproc.filter_maf_interaction` and `interaction_half_mask`.
  - **Command line** `python -m localqtl` / `localqtl` with tensorQTL's modes (`cis`,
    `cis_nominal`, `cis_independent`, `cis_susie`, `trans`, `trans_susie`), arguments and output
    file names, plus `--haplotypes`, `--la_format`, `--la_source`, `--la_cache_dir`,
    `--use_posterior`, `--ancestry_model`, `--min_ancestry_ac`, `--device`, `--perm_chunk` and
    `--localqtl_names`. Output columns use tensorQTL's spelling unless `--localqtl_names` is given.
- API stubs for the planned `localqtl.annotate.alphagenome` and `localqtl.annotate.igvf`
  modules (they raise `NotImplementedError`).
- Continuous integration: tests on Python 3.11–3.13 and a docs build on every push and pull
  request; tagged releases build, publish to PyPI (trusted publishing) and create a GitHub
  release whose body is this file's section.

### Changed

- **Degrees of freedom.** `Residualizer.k_eff` now counts the centring (intercept) degree of
  freedom in addition to the covariate rank, and every kernel uses `dof = n - k_eff - p`. Standard
  errors from the batched kernels were previously too small by a factor of
  `sqrt((n - k - 1) / (n - k - 2))` and inconsistent with the correlation-based p-values.
- **No-covariate runs fit an intercept.** With `covariates_df=None` the residualizer has rank 0
  and centres every matrix, instead of fitting through the origin.
- **Interaction models match full OLS.** The `g:i` column is built from the raw genotypes and the
  interaction main effect is residualized against the covariates (Frisch–Waugh), in the nominal,
  permutation and independent cores; previously the product used the residualized genotype,
  which biased all interaction coefficients whenever covariates were present. `map_nominal`
  with a local-ancestry panel now keeps the ancestry main effects in the GxC model, as the
  permutation and independent scans already did.
- `ancestry_model` is validated everywhere (`'main'`, `'interaction'`, `'dosage'`); misspelled
  values no longer run the main model silently.
- `map_nominal_loci` padded `position` by the window twice; it now pads once.
- `impute_mean_and_filter` drops rows that become monomorphic only after mean imputation.
- `read_phenotype_bed` keeps the case of sample IDs (they were lowercased).
- The dosage-model Parquet schema gained the per-ancestry allele-count columns (appended after
  the pairwise-difference block). The `af` column is documented as the in-sample frequency of
  the counted (`alt`) allele.
- Build backend: `poetry-dynamic-versioning` (the version is read from the git tag, also for
  `pip install` from a checkout).

### Fixed

- `map_independent`: the grouped core referenced an undefined name; the forward pass could loop
  forever when a conditioned variant was re-selected; the Beta fit could hang on degenerate
  permutation sets (it now returns NaN and callers fall back to `pval_perm`); a NaN `true_dof`
  with `beta_approx=False` crashed the integer result buffer.
- `PgenReader(select_samples=...)` passed a Python list to pgenlib, which rejected it.
- Tests that asserted group columns on an ungrouped run, and a permutation test that hung on
  the forward-pass loop above, were corrected.

## [0.1.4] - 2026-03-28

- SuSiE fine-mapping (`localqtl.susie`), COLOC colocalization (`localqtl.coloc`) and
  `interaction_covariate` (genotype-by-covariate) scans; cis post-processing helpers
  `annotate_with_susie`, `annotate_with_coloc`, `annotate_ancestry_difference`.
- Ancestry-specific effect columns for the GxH `ancestry_model="interaction"` scan.
- Lazy GPU array/dataframe backends (CuPy, cuDF) with CPU fallback; optional `gpu` extra.
- rfmix-reader `>= 0.6` per-chromosome `LocalAncestryPanel` (`RFMixReader`, `FlareReader`,
  `open_panel`).
- p-values computed with SciPy on CPU to avoid clamping; degrees-of-freedom and AMP fixes in
  the permutation kernels.

## [0.1.3] - 2025-11-12

- Read the Docs configuration and refreshed documentation; `nperm` defaults to no permutations.
- Fixed the degrees-of-freedom call and the `alpha_perm` naming in the permutation scan.

## [0.1.2] - 2025-11-03

- Permutation kernels without quadratic memory, shared GPU permutations across phenotypes,
  covariate and permutation reuse in `map_independent`, batch progress logging.
- `_torch_utils` device helpers; `preload_haplotypes` for every cis mapper.

## [0.1.1] - 2025-11-01

- `pandas-plink-ng` replaces the git install of `pandas-plink`; optional haplotype preloading
  to torch.

## [0.1.0] - 2025-11-01

- First release: tensorQTL-style cis-QTL mapping (`map_nominal`, `map_permutations`,
  `map_independent`, `CisMapper`) in PyTorch with local-ancestry haplotype input from RFMix
  (`InputGeneratorCisWithHaps`), PLINK 1/2 readers, streaming Parquet sinks and pure-Python
  q-values.

[Unreleased]: https://github.com/heart-gen/localQTL/compare/v0.4.4...HEAD
[0.4.4]: https://github.com/heart-gen/localQTL/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/heart-gen/localQTL/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/heart-gen/localQTL/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/heart-gen/localQTL/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/heart-gen/localQTL/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/heart-gen/localQTL/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/heart-gen/localQTL/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/heart-gen/localQTL/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/heart-gen/localQTL/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/heart-gen/localQTL/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/heart-gen/localQTL/compare/v0.1.0-dev...v0.1.1
[0.1.0]: https://github.com/heart-gen/localQTL/releases/tag/v0.1.0-dev
