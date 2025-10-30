import numpy as np
import torch

from src.localqtl.cis.independent import map_independent
from src/localqtl.cis.permutations import map_permutations
from src.localqtl.utils import SimpleLogger


def test_map_independent_identifies_forward_backward_hits(toy_data):
    torch.manual_seed(0)
    base = map_permutations(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=4,
        seed=321,
        logger=SimpleLogger(verbose=False),
    )

    cis_df = base.copy()
    cis_df["qval"] = np.linspace(0.01, 0.03, len(cis_df))

    result = map_independent(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        cis_df=cis_df,
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=3,
        seed=123,
        logger=SimpleLogger(verbose=False),
    )

    assert not result.empty
    assert {"phenotype_id", "variant_id", "rank", "pval_beta"}.issubset(result.columns)
    # ranks should start at 1 for each phenotype
    grouped = result.groupby("phenotype_id")["rank"]
    for ranks in grouped:
        assert ranks.iloc[0] == 1
