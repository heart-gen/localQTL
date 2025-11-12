import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys
import types

try:  # pragma: no cover - optional dependency stub for tests
    import rfmix_reader  # type: ignore
except ImportError:  # pragma: no cover - exercised when dependency missing
    rfmix_reader = types.ModuleType("rfmix_reader")  # type: ignore[assignment]
    sys.modules.setdefault("rfmix_reader", rfmix_reader)


def _missing(*_args, **_kwargs):  # pragma: no cover - helper for optional dep
    raise ImportError("rfmix_reader optional dependency is not installed for tests")


for _name in ("read_rfmix", "read_flare"):
    if not hasattr(rfmix_reader, _name):  # type: ignore[attr-defined]
        setattr(rfmix_reader, _name, _missing)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from localqtl.cis.independent import map_independent
from localqtl.utils import SimpleLogger


def _build_synthetic_dataset(n_samples: int = 48,
                             n_variants: int = 12,
                             n_phenotypes: int = 4,
                             seed: int = 1337
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(n_samples)]
    variant_ids = [f"var{i}" for i in range(n_variants)]
    phenotype_ids = [f"pheno{i}" for i in range(n_phenotypes)]

    genotype = rng.normal(size=(n_variants, n_samples)).astype(np.float32)
    genotype_df = pd.DataFrame(genotype, index=variant_ids, columns=samples)

    variant_df = pd.DataFrame({
        "chrom": ["chr1"] * n_variants,
        "pos": np.arange(n_variants, dtype=np.int32) * 1000 + 100,
    }, index=variant_ids)

    phenotype_matrix = []
    cis_rows = []
    phenotype_pos_rows = []
    for i, pid in enumerate(phenotype_ids):
        lead_idx = i % n_variants
        effect = 0.8 + 0.1 * rng.random()
        noise = rng.normal(scale=0.5, size=n_samples).astype(np.float32)
        phenotype = effect * genotype[lead_idx] + noise
        phenotype_matrix.append(phenotype)
        cis_rows.append({
            "phenotype_id": pid,
            "variant_id": variant_ids[lead_idx],
            "pval_beta": 1e-4,
            "qval": 0.01,
            "pval_perm": 1e-4,
            "pval_nominal": 1e-4,
            "beta_shape1": 2.0,
            "beta_shape2": 2.0,
            "ma_samples": 10,
            "ma_count": 20.0,
            "af": 0.3,
            "num_var": n_variants,
            "slope": effect,
            "slope_se": 0.1,
            "tstat": 5.0,
            "r2_nominal": 0.25,
            "true_dof": n_samples - 2,
            "pval_true_dof": 1e-4,
            "start_distance": 0,
            "end_distance": 0,
        })
        phenotype_pos_rows.append({
            "chr": "chr1",
            "start": 100_000 + i * 10_000,
            "end": 100_000 + i * 10_000 + 100,
        })

    phenotype_df = pd.DataFrame(phenotype_matrix, index=phenotype_ids, columns=samples)
    phenotype_pos_df = pd.DataFrame(phenotype_pos_rows, index=phenotype_ids)
    cis_df = pd.DataFrame(cis_rows)

    covariate = rng.normal(size=(n_samples, 1)).astype(np.float32)
    covariates_df = pd.DataFrame(covariate, index=samples, columns=["cov1"])

    return genotype_df, variant_df, cis_df, phenotype_df, phenotype_pos_df, covariates_df


def test_bench_independent():
    genotype_df, variant_df, cis_df, phenotype_df, phenotype_pos_df, covariates_df = _build_synthetic_dataset()

    logger = SimpleLogger(verbose=False)

    def run_map(nperm_stage1: int | None) -> pd.DataFrame:
        start = time.perf_counter()
        result = map_independent(
            genotype_df=genotype_df,
            variant_df=variant_df,
            cis_df=cis_df,
            phenotype_df=phenotype_df,
            phenotype_pos_df=phenotype_pos_df,
            covariates_df=covariates_df,
            haplotypes=None,
            loci_df=None,
            group_s=None,
            maf_threshold=0.0,
            fdr=0.05,
            nperm=64,
            window=200_000,
            missing=-9.0,
            random_tiebreak=False,
            device="cpu",
            beta_approx=True,
            perm_chunk=16,
            seed=2024,
            logger=logger,
            verbose=False,
            preload_haplotypes=True,
            tensorqtl_flavor=False,
            perm_indices_mode="generator",
            mixed_precision="off",
            nperm_stage1=nperm_stage1,
        )
        elapsed = time.perf_counter() - start
        print(f"chr1 elapsed {elapsed:.3f}s (nperm_stage1={nperm_stage1})")
        return result.sort_values(["phenotype_id", "rank"]).reset_index(drop=True)

    baseline = run_map(nperm_stage1=0)
    staged = run_map(nperm_stage1=8)
    staged_repeat = run_map(nperm_stage1=8)

    pd.testing.assert_frame_equal(staged, staged_repeat, check_dtype=False)

    assert baseline[["phenotype_id", "variant_id", "rank"]].equals(
        staged[["phenotype_id", "variant_id", "rank"]]
    )
    max_delta = (staged["pval_perm"] - baseline["pval_perm"]).abs().max()
    assert max_delta <= 5e-4
