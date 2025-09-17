import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append("src")
import haplotypeio as hio


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def fake_rfmix_data_2ancestries(monkeypatch):
    loci = pd.DataFrame({"chromosome":["1","1"], "physical_position":[100,200]})
    g_anc = pd.DataFrame({"sample_id":["s1","s2"], "chrom":["1","1"]})
    admix = np.array([
        [[1,0],[0,1]],  # variant 1: 2 samples x 2 ancestries
        [[0,1],[1,0]]   # variant 2
    ])  # shape (variants, samples, ancestries)

    def fake_read_rfmix(prefix_path, binary_dir, verbose):
        return loci, g_anc, admix

    monkeypatch.setattr(hio, "read_rfmix", fake_read_rfmix)
    return loci, g_anc, admix


@pytest.fixture
def fake_rfmix_data_3ancestries(monkeypatch):
    loci = pd.DataFrame({"chromosome":["1","1"], "physical_position":[100,200]})
    g_anc = pd.DataFrame({"sample_id":["s1","s2"], "chrom":["1","1"]})
    admix = np.zeros((2, 2, 3))  # 2 variants, 2 samples, 3 ancestries
    admix[0,0,0] = 1
    admix[0,1,1] = 1
    admix[1,0,2] = 1
    admix[1,1,0] = 1

    def fake_read_rfmix(prefix_path, binary_dir, verbose):
        return loci, g_anc, admix

    monkeypatch.setattr(hio, "read_rfmix", fake_read_rfmix)
    return loci, g_anc, admix


# ----------------------------
# Tests
# ----------------------------
def test_rfmixreader_two_ancestries(fake_rfmix_data_2ancestries):
    reader = hio.RFMixReader("dummy_prefix", verbose=False)
    assert reader.n_pops == 2
    assert isinstance(reader.loci_df, pd.DataFrame)
    assert reader.haplotypes.shape[2] == 1  # reduced to single ancestry column (A0)


def test_rfmixreader_three_ancestries(fake_rfmix_data_3ancestries):
    reader = hio.RFMixReader("dummy_prefix", verbose=False)
    assert reader.n_pops == 3
    assert isinstance(reader.loci_df, pd.DataFrame)
    assert set(reader.loci_df["ancestry"].unique()) == {0,1,2}
    assert reader.haplotypes.shape == (2, 2, 3)  # unchanged


def test_inputgeneratorciswithhaps_yields_batches(fake_rfmix_data_2ancestries):
    loci, g_anc, admix = fake_rfmix_data_2ancestries
    haplotypes = admix
    loci_df = pd.DataFrame({"chrom":["1","1"],"pos":[100,200]})
    genotype_df = pd.DataFrame([[0,1],[1,2]], index=["v1","v2"], columns=["s1","s2"])
    variant_df = pd.DataFrame({"chrom":["1","1"],"pos":[100,200]}, index=["v1","v2"])
    phenotype_df = pd.DataFrame([[1,2]], index=["g1"], columns=["s1","s2"])
    phenotype_pos_df = pd.DataFrame({"chr":["1"],"pos":[150]}, index=["g1"])
    
    gen = hio.InputGeneratorCisWithHaps(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        haplotypes=haplotypes, loci_df=loci_df
    )
    batches = list(gen.generate_data())
    assert len(batches) == 1
    p, G, v_idx, H, pid = batches[0]
    assert H.shape[1] == 2
    assert pid == "g1"
