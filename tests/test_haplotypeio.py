import numpy as np
import pandas as pd
import pytest
import sys

# ensure we import from the local package
sys.path.append("src")

import localqtl.haplotypeio as hio
from localqtl.haplotypeio import InputGeneratorCisWithHaps

############################
# Fixtures for RFMixReader
############################

@pytest.fixture
def fake_rfmix_data_2ancestries(monkeypatch):
    """
    Mock read_rfmix() -> (loci_df_like, g_anc, admix)
    where admix has 2 ancestries, shape (variants, samples, 2).
    """
    loci = pd.DataFrame({
        "chromosome": ["1", "1", "2"],
        "physical_position": [100, 200, 300],
    })
    g_anc = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "chrom": ["1", "1", "2", "2"],
    })
    # admix[v, s, a] = simple pattern v + 10*s + 100*a
    n_var = loci.shape[0]    # 3 loci
    n_s   = g_anc.shape[0]   # 4 samples
    n_pop = 2
    admix = np.zeros((n_var, n_s, n_pop), dtype=float)
    for v in range(n_var):
        for s in range(n_s):
            for a in range(n_pop):
                admix[v, s, a] = v + 10*s + 100*a

    def fake_read_rfmix(prefix_path, binary_dir, verbose):
        return loci, g_anc, admix

    monkeypatch.setattr(hio, "read_rfmix", fake_read_rfmix)
    return loci, g_anc, admix


@pytest.fixture
def fake_rfmix_data_3ancestries(monkeypatch):
    """
    Mock read_rfmix() but with 3 ancestries.
    """
    loci = pd.DataFrame({
        "chromosome": ["1", "1", "2"],
        "physical_position": [100, 200, 300],
    })
    g_anc = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "chrom": ["1", "2"],
    })
    n_var = loci.shape[0]   # 3
    n_s   = g_anc.shape[0]  # 2
    n_pop = 3
    admix = np.zeros((n_var, n_s, n_pop), dtype=float)
    # ancestry0 ~ v + s
    # ancestry1 ~ v + 10*s
    # ancestry2 ~ v + 100*s
    for v in range(n_var):
        for s in range(n_s):
            admix[v, s, 0] = v + s
            admix[v, s, 1] = v + 10*s
            admix[v, s, 2] = v + 100*s

    def fake_read_rfmix(prefix_path, binary_dir, verbose):
        return loci, g_anc, admix

    monkeypatch.setattr(hio, "read_rfmix", fake_read_rfmix)
    return loci, g_anc, admix


##################################
# Tests for RFMixReader behavior
##################################

def test_rfmixreader_two_ancestries(fake_rfmix_data_2ancestries):
    """
    n_pops == 2 branch:
      - haplotypes should become admix[:, :, [0]]
        (only first ancestry retained)
      - loci_df should exist and be a pandas DataFrame
    """
    reader = hio.RFMixReader("dummy_prefix", verbose=False)

    # n_pops detected correctly
    assert reader.n_pops == 2

    # haplotypes collapsed to first ancestry only -> shape (loci, samples, 1)
    assert reader.haplotypes.shape[2] == 1
    assert reader.haplotypes.shape[0] == 3  # loci
    assert isinstance(reader.loci_df, pd.DataFrame)

    # sample_ids from g_anc
    assert reader.sample_ids == ["s1", "s2", "s3", "s4"]

    # loci_df index should be hap IDs (ends with _A0)
    assert all(h.endswith("_A0") for h in reader.loci_df.index.tolist())


def test_rfmixreader_three_ancestries(fake_rfmix_data_3ancestries):
    """
    n_pops > 2 branch:
      - haplotypes should be full admix array
      - loci_df should stack ancestries (A0, A1, A2) and have 'ancestry' column
    """
    reader = hio.RFMixReader("dummy_prefix", verbose=False)

    assert reader.n_pops == 3
    # unchanged shape (loci, samples, ancestries)
    assert reader.haplotypes.shape == (3, 2, 3)
    assert isinstance(reader.loci_df, pd.DataFrame)

    # should have ancestry column with {0,1,2}
    assert set(reader.loci_df["ancestry"].unique()) == {0, 1, 2}

    # hap IDs in index look like chr_pos_A{k}
    assert any(h.endswith("_A2") for h in reader.loci_df.index.tolist())


###########################################################
# Tests for InputGeneratorCisWithHaps using toy fixtures
###########################################################

def test_inputgeneratorciswithhaps_batches_match_cis(toy_hap_data_2anc):
    """
    Use the shared toy data fixture:
      - genotype_df / variant_df / phenotype_df / phenotype_pos_df
      - haplotypes (loci x samples x 2)
      - loci_df
    Check that:
      - generator yields 1 batch (because 1 phenotype)
      - shapes line up
      - haplotype slice H corresponds to rows v_idx
      - interpolation ran (i.e. no NaNs remaining in H where there were partial NaNs)
    """
    geno_df      = toy_hap_data_2anc["genotype_df"]
    var_df       = toy_hap_data_2anc["variant_df"]
    pheno_df     = toy_hap_data_2anc["phenotype_df"]
    pheno_pos_df = toy_hap_data_2anc["phenotype_pos_df"]
    hap          = toy_hap_data_2anc["haplotypes"]
    loci_df      = toy_hap_data_2anc["loci_df"]

    gen = InputGeneratorCisWithHaps(
        geno_df,
        var_df,
        pheno_df,
        pheno_pos_df,
        haplotypes=hap,
        loci_df=loci_df,
        on_the_fly_impute=True,
        window=200,   # wide enough to grab cis around 150/900 etc.
    )

    # materialize generator to list so background wrapper fully runs
    batches = list(gen.generate_data(verbose=True))

    # We expect exactly 1 phenotype ("g1") in toy_data
    assert len(batches) == 1

    p, G, v_idx, H, pid = batches[0]

    # pid should be the phenotype ID from toy data
    assert pid == "g1"

    # p is (samples,), G is (variants_in_window x samples),
    # v_idx is indices into variant_df/genotype_df,
    # H is (variants_in_window x samples x ancestries)
    assert p.shape[0] == geno_df.shape[1]            # samples
    assert G.shape[1] == geno_df.shape[1]            # samples
    assert H.shape[1] == geno_df.shape[1]            # samples
    assert H.shape[2] == 2                           # ancestries in toy_hap_data_2anc

    # v_idx should be increasing indices into variant_df/genotype_df rows
    assert np.all(np.diff(v_idx) >= 0)

    # H rows should correspond exactly to those v_idx rows from the input haplotype cube
    # (after interpolation). So shape[0] should equal len(v_idx).
    assert H.shape[0] == len(v_idx)

    # After interpolation, NaNs in partially-missing columns should be filled
    assert not np.isnan(H).any()


def test_interpolate_block_handles_nans(toy_hap_data_3anc):
    """
    Directly unit test _interpolate_block on a small block
    with NaNs in the middle of a column. Should linearly
    interpolate along loci dimension and round().
    """
    block = np.array([
        [[0.0,  10.0, 100.0],
         [1.0,  11.0, 101.0]],   # locus 0
        [[np.nan, 20.0, 200.0],
         [np.nan, 21.0, 201.0]], # locus 1 (NaNs in ancestry0 for both samples)
        [[2.0,  30.0, 300.0],
         [3.0,  31.0, 301.0]],   # locus 2
    ], dtype=float)
    # block shape: (3 loci, 2 samples, 3 ancestries)

    imputed = InputGeneratorCisWithHaps._interpolate_block(block)

    # NaNs in ancestry0 (0th ancestry channel) should be filled by linear interpolation
    # Between 0.0 (locus0,sample0,anc0) and 2.0 (locus2,sample0,anc0),
    # locus1 should become ~1.0 then rounded to int -> 1
    assert imputed[1,0,0] == 1

    # Between 1.0 and 3.0 for sample1 anc0 -> should become ~2 then rounded to int -> 2
    assert imputed[1,1,0] == 2

    # entries that were not NaN should remain the same
    assert imputed[0,0,1] == 10.0
    assert imputed[2,1,2] == 301.0
