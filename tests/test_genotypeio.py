import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append("src")  # adjust to where genotypeio.py lives
from genotypeio import _impute_mean, get_cis_ranges, InputGeneratorCis

def test_impute_mean_basic():
    g = np.array([[1.0, -9.0, 2.0]], dtype=np.float32)
    _impute_mean(g, missing=-9)
    assert np.isclose(g[0, 1], (1.0 + 2.0) / 2)

def test_get_cis_ranges():
    phenotype_pos_df = pd.DataFrame({
        "chr": ["1"], "start": [100], "end": [100]
    }, index=["pheno1"])
    chr_variant_dfs = {
        "1": pd.DataFrame({"pos": [50, 100, 200], "index": [0, 1, 2]})
    }
    cis_ranges, drop_ids = get_cis_ranges(phenotype_pos_df, chr_variant_dfs, window=50)
    assert "pheno1" in cis_ranges
    assert drop_ids == []

def test_input_generator_cis_yields_batches():
    genotype_df = pd.DataFrame([[0,1,2],[1,2,0]], index=["v1","v2"], columns=["s1","s2"])
    variant_df = pd.DataFrame({"chrom":["1","1"],"pos":[100,200]}, index=["v1","v2"])
    phenotype_df = pd.DataFrame([[1,2]], index=["g1"], columns=["s1","s2"])
    phenotype_pos_df = pd.DataFrame({"chr":["1"],"pos":[150]}, index=["g1"])
    
    gen = InputGeneratorCis(genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=100)
    batches = list(gen.generate_data(verbose=True))
    assert len(batches) == 1
    p, G, v_idx, pid = batches[0]
    assert pid == "g1"
    assert G.shape[1] == 2
