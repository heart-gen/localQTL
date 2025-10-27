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

def _make_loci_df(chroms, positions, n_pops):
    """
    chroms: list[str] length = n_loci
    positions: list[int] length = n_loci
    n_pops: number of ancestries

    Returns:
      loci_df (pd.DataFrame):
        index: hap IDs like "1_100_A0"
        columns: chrom, pos, ancestry, index
        'index' is 0..n_loci-1 for pop0, then 0..n_loci-1 for pop1, etc.
        (this is similar in spirit to haplotypeio.RFMixReader for n_pops>2,
         and a simplified version for n_pops==2)
    """
    records = []
    n_loci = len(chroms)
    for anc in range(n_pops):
        for i in range(n_loci):
            hap_id = f"{chroms[i]}_{positions[i]}_A{anc}"
            records.append({
                "hap": hap_id,
                "chrom": chroms[i],
                "pos": positions[i],
                "ancestry": anc,
                # This matches the pattern where index can be offset by ancestry
                "index": anc * n_loci + i,
            })
    loci_df = pd.DataFrame.from_records(records).set_index("hap")
    return loci_df


@pytest.fixture
def toy_hap_data_2anc(toy_data):
    """
    Build small, deterministic haplotype/local ancestry data for 2 ancestries.
    Mirrors toy_data's variant_df rows (7 loci across chr1/chr2).

    We create:
      haplotypes: shape (7 loci, 4 samples, 2 ancestries)
      loci_df: metadata with hap IDs per (locus, ancestry)
      plus we forward genotype_df/etc from toy_data so tests
      can directly build InputGeneratorCisWithHaps.
    """
    genotype_df      = toy_data["genotype_df"]
    variant_df       = toy_data["variant_df"]
    phenotype_df     = toy_data["phenotype_df"]
    phenotype_pos_df = toy_data["phenotype_pos_df"]

    # loci: same order as variant_df rows
    chroms = variant_df["chrom"].tolist()   # ["1","1","1","1","1","2","2"]
    positions = variant_df["pos"].tolist()  # [100,200,300,800,900,150,250]

    n_loci    = len(positions)  # 7
    n_samples = genotype_df.shape[1]  # should be 4
    n_pops    = 2  # two ancestries

    # Shape: (loci, samples, ancestries)
    # We'll make a simple deterministic pattern:
    # ancestry0 = locus_index + sample_index
    # ancestry1 = locus_index + 10*sample_index
    hap_anc0 = np.zeros((n_loci, n_samples), dtype=float)
    hap_anc1 = np.zeros((n_loci, n_samples), dtype=float)
    for i in range(n_loci):
        for s in range(n_samples):
            hap_anc0[i, s] = i + s       # e.g. 0+0,0+1,0+2,0+3; 1+0,1+1,...
            hap_anc1[i, s] = i + 10*s    # e.g. 0+0,0+10,0+20,0+30; 1+0,...

    hap = np.stack([hap_anc0, hap_anc1], axis=2)  # (loci, samples, 2)

    # inject some NaNs for interpolation test:
    # say locus 2, sample 1, ancestry0 is missing
    hap[2, 1, 0] = np.nan
    # also locus 5, sample 3, ancestry1 missing
    hap[5, 3, 1] = np.nan

    loci_df = _make_loci_df(chroms, positions, n_pops=2)

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "haplotypes": hap,     # numpy ndarray (loci x samples x 2)
        "loci_df": loci_df,    # pandas DataFrame
    }


@pytest.fixture
def toy_hap_data_3anc(toy_data):
    """
    Same idea as toy_hap_data_2anc, but now with 3 ancestries.
    Used to test n_pops > 2 behavior and interpolation.

    We don't strictly need phenotype_df here, but we include it for symmetry.
    """
    genotype_df      = toy_data["genotype_df"]
    variant_df       = toy_data["variant_df"]
    phenotype_df     = toy_data["phenotype_df"]
    phenotype_pos_df = toy_data["phenotype_pos_df"]

    chroms = variant_df["chrom"].tolist()
    positions = variant_df["pos"].tolist()

    n_loci    = len(positions)
    n_samples = genotype_df.shape[1]  # 4
    n_pops    = 3

    # ancestry0 = i + s
    # ancestry1 = i + 10*s
    # ancestry2 = i + 100*s
    hap_ancs = []
    for anc in range(n_pops):
        anc_mat = np.zeros((n_loci, n_samples), dtype=float)
        for i in range(n_loci):
            for s in range(n_samples):
                anc_mat[i, s] = i + (10**anc)*s
        hap_ancs.append(anc_mat)
    hap = np.stack(hap_ancs, axis=2)  # (loci, samples, 3)

    # Add a couple NaNs to test interpolation
    hap[1, 2, 0] = np.nan  # ancestry0 missing at (locus1, sample2)
    hap[4, 0, 2] = np.nan  # ancestry2 missing at (locus4, sample0)

    loci_df = _make_loci_df(chroms, positions, n_pops=3)

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "haplotypes": hap,     # numpy ndarray (loci x samples x 3)
        "loci_df": loci_df,    # pandas DataFrame with ancestry column 0..2
    }

