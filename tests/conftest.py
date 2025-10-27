import numpy as np
import pandas as pd
import pytest

############################
# Core synthetic test data #
############################

def make_test_data():
    # ---------- Samples ----------
    samples = ["S1", "S2", "S3", "S4"]

    # ---------- Variant metadata ----------
    variant_index = [
        "v1_chr1_100",
        "v2_chr1_200",
        "v3_chr1_300",
        "v4_chr1_800",
        "v5_chr1_900",
        "v1_chr2_150",
        "v2_chr2_250",
        "v3_chr2_400",
    ]

    chroms = ["1","1","1","1","1","2","2","2"]
    poses  = [100, 200, 300, 800, 900, 150, 250, 400]

    variant_df = pd.DataFrame({
        "chrom": chroms,
        "pos": poses,
    }, index=variant_index)

    # ---------- Genotypes ----------
    # shape: variants x samples (8 x 4)
    # We'll deliberately include a missing code -9 in some rows
    genotype_mat = np.array([
        [0, 1, 2, 0],   # v1_chr1_100
        [1, 1, 2, 2],   # v2_chr1_200
        [2, 2, 2,-9],   # v3_chr1_300  <-- has missing
        [0, 0, 1, 1],   # v4_chr1_800
        [2, 1, 1, 0],   # v5_chr1_900
        [0, 0, 0, 0],   # v1_chr2_150
        [1, 2, 1, 1],   # v2_chr2_250
        [2, 2, 1, 1],   # v3_chr2_400
    ], dtype=np.float32)

    genotype_df = pd.DataFrame(
        genotype_mat,
        index=variant_index,
        columns=samples
    )

    # ---------- Phenotypes + positions ----------
    # We'll include one constant phenotype that should get dropped.
    phenotype_index = ["geneA", "geneB", "geneC", "geneConst"]

    phenotype_mat = np.array([
        [ 10.0,  11.0,  12.0,  13.0],  # geneA  (varies)
        [  5.0,   5.5,   6.0,   6.5],  # geneB  (varies)
        [100.0, 110.0, 105.0, 120.0],  # geneC  (varies)
        [ 42.0,  42.0,  42.0,  42.0],  # geneConst (constant -> should drop)
    ], dtype=np.float32)

    phenotype_df = pd.DataFrame(
        phenotype_mat,
        index=phenotype_index,
        columns=samples
    )

    # phenotype_pos_df in "start/end" format
    phenotype_pos_df = pd.DataFrame({
        "chr":  ["1",  "1",  "2",  "1"],
        "start":[180, 850, 260, 500],
        "end":  [180, 900, 260, 500],
    }, index=phenotype_index)

    return genotype_df, variant_df, phenotype_df, phenotype_pos_df


###########################
# Pytest fixtures version #
###########################

@pytest.fixture
def toy_data():
    """
    Returns a dict with all basic toy dataframes needed to test genotypeio.
    """
    (genotype_df,
     variant_df,
     phenotype_df,
     phenotype_pos_df) = make_test_data()

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "samples": list(genotype_df.columns),
    }

