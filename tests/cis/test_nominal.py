import numpy as np
import torch

from src.localqtl.cis.nominal import map_nominal
from src.localqtl.utils import SimpleLogger


def test_map_nominal_returns_all_pairs(toy_data):
    torch.manual_seed(0)
    result = map_nominal(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=None,
        device="cpu",
        nperm=3,
        logger=SimpleLogger(verbose=False),
    )

    assert {"phenotype_id", "variant_id", "pval_nominal", "perm_max_r2"}.issubset(result.columns)
    # geneConst should be dropped; the remaining phenotypes must appear in the output
    expected_ids = {pid for pid in toy_data["phenotype_df"].index if pid != "geneConst"}
    assert set(result["phenotype_id"].unique()) == expected_ids
    assert ((result["pval_nominal"] >= 0.0) & (result["pval_nominal"] <= 1.0)).all()
    assert np.isfinite(result["perm_max_r2"]).all()
