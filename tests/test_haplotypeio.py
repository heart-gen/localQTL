import numpy as np
import pandas as pd
import pytest
import sys
import torch

# ensure we import from the local package
sys.path.append("src")

import localqtl.haplotypeio as hio
from localqtl.haplotypeio import InputGeneratorCisWithHaps

############################
# Fixtures for RFMixReader
############################

def _legacy_dataset(n_pops, n_samples, seed=0):
    """A small rfmix_reader Dataset built from a legacy (loci, g_anc, admix) triple."""
    from rfmix_reader import from_legacy

    loci = pd.DataFrame({"chromosome": ["1", "1", "2"], "physical_position": [100, 200, 300]})
    g_anc = pd.DataFrame({"sample_id": [f"s{i + 1}" for i in range(n_samples)]})
    pops = ["EUR", "AFR", "EAS"][:n_pops]
    for k, pop in enumerate(pops):
        g_anc[pop] = np.linspace(0.2, 0.8, n_samples) if k == 0 else (1 - np.linspace(0.2, 0.8, n_samples)) / max(n_pops - 1, 1)
    g_anc["chrom"] = "1"
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, n_pops, size=(3, n_samples, 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(n_pops)], axis=-1).astype(np.int8)
    return from_legacy(loci, g_anc, admix), admix


@pytest.fixture
def fake_rfmix_data_2ancestries(monkeypatch):
    ds, admix = _legacy_dataset(2, 4)
    monkeypatch.setattr(hio, "open_rfmix", lambda *a, **k: ds, raising=False)
    import rfmix_reader
    monkeypatch.setattr(rfmix_reader, "open_rfmix", lambda *a, **k: ds)
    return ds, admix


@pytest.fixture
def fake_rfmix_data_3ancestries(monkeypatch):
    ds, admix = _legacy_dataset(3, 2)
    import rfmix_reader
    monkeypatch.setattr(rfmix_reader, "open_rfmix", lambda *a, **k: ds)
    return ds, admix


##################################
# Tests for RFMixReader behavior
##################################

def test_rfmixreader_two_ancestries(fake_rfmix_data_2ancestries):
    """
    n_pops == 2: only the first ancestry channel is exposed (the second is
    2 - first); loci_df keeps the legacy hap-indexed layout.
    """
    ds, admix = fake_rfmix_data_2ancestries
    reader = hio.RFMixReader("dummy_prefix", verbose=False)

    assert reader.n_pops == 2 and reader.n_channels == 1
    assert reader.haplotypes.shape == (3, 4, 1)
    assert reader.sample_ids == ["s1", "s2", "s3", "s4"]
    assert isinstance(reader.loci_df, pd.DataFrame)
    assert all(h.endswith("_A0") for h in reader.loci_df.index.tolist())
    assert reader.loci["chrom"].tolist() == ["1", "1", "2"] and reader.loci["pos"].tolist() == [100, 200, 300]

    H = reader.load_haplotypes()
    assert H.shape == (3, 4, 1) and H.dtype == np.int8
    np.testing.assert_array_equal(H[:, :, 0], admix[:, :, 0])
    np.testing.assert_array_equal(reader.chrom_counts("1"), admix[:2, :, [0]])
    np.testing.assert_array_equal(reader.chrom_counts("chr2"), admix[2:, :, [0]])
    np.testing.assert_array_equal(reader.locus_index("1", [100, 150, 200, 250]), [0, -1, 1, -1])
    np.testing.assert_array_equal(reader.locus_index("1", [150, 250], method="nearest"), [0, 1])
    np.testing.assert_array_equal(reader.locus_index("9", [1]), [-1])


def test_rfmixreader_three_ancestries(fake_rfmix_data_3ancestries):
    ds, admix = fake_rfmix_data_3ancestries
    reader = hio.RFMixReader("dummy_prefix", verbose=False)

    assert reader.n_pops == 3 and reader.n_channels == 3
    assert reader.haplotypes.shape == (3, 2, 3)
    assert set(reader.loci_df["ancestry"].unique()) == {0, 1, 2}
    assert any(h.endswith("_A2") for h in reader.loci_df.index.tolist())
    np.testing.assert_array_equal(reader.load_haplotypes(), admix)


def test_panel_select_samples_and_exclude_chrs(fake_rfmix_data_2ancestries):
    ds, admix = fake_rfmix_data_2ancestries
    panel = hio.LocalAncestryPanel(ds, select_samples=["s3", "s1"], exclude_chrs=["chr2"])
    assert panel.sample_ids == ["s3", "s1"] and panel.chromosomes == ["1"]
    np.testing.assert_array_equal(panel.load_haplotypes()[:, :, 0], admix[:2][:, [2, 0], 0])
    with pytest.raises(ValueError, match="not in the local ancestry"):
        hio.LocalAncestryPanel(ds, select_samples=["s1", "nope"])


def _panel_for_toy(toy, n_pops=2, seed=0):
    """Valid diploid counts on the toy variant grid, as a LocalAncestryPanel."""
    from rfmix_reader import from_legacy

    var_df = toy["variant_df"]
    samples = list(toy["genotype_df"].columns)
    loci = pd.DataFrame({"chromosome": var_df["chrom"].astype(str).values,
                         "physical_position": var_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": samples})
    for k in range(n_pops):
        g_anc[f"P{k}"] = 1.0 / n_pops
    g_anc["chrom"] = "1"
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, n_pops, size=(len(loci), len(samples), 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(n_pops)], axis=-1).astype(np.int8)
    return hio.LocalAncestryPanel(from_legacy(loci, g_anc, admix)), admix


def test_generator_panel_mode_matches_direct_indexing(toy_hap_data_2anc):
    """Panel mode: H is the per-chromosome count array indexed by the SNP's segment."""
    toy = toy_hap_data_2anc
    panel, admix = _panel_for_toy(toy)
    gen = InputGeneratorCisWithHaps(
        toy["genotype_df"], toy["variant_df"], toy["phenotype_df"], toy["phenotype_pos_df"],
        haplotypes=panel, window=200,
    )
    assert gen.n_unmapped_variants == 0
    batches = list(gen.generate_data())
    assert len(batches) == 3
    for p, G, v_idx, H, pid in batches:
        assert isinstance(H, np.ndarray) and H.dtype == np.float32
        assert H.shape == (len(v_idx), toy["genotype_df"].shape[1], 1)
        np.testing.assert_array_equal(H[:, :, 0], admix[v_idx, :, 0])   # loci == variant grid here
        assert not np.isnan(H).any()


def test_generator_panel_mode_segment_lookup_and_missing(toy_hap_data_2anc):
    """A coarser panel (fewer loci than SNPs) maps SNPs to the covering segment;
    SNPs outside every segment are dropped; missing calls are interpolated."""
    from rfmix_reader import from_legacy

    toy = toy_hap_data_2anc
    var_df = toy["variant_df"]                      # chr1: 100,200,300,800,900 ; chr2: 150,250
    samples = list(toy["genotype_df"].columns)
    loci = pd.DataFrame({"chromosome": ["1", "1", "2"], "physical_position": [100, 800, 150]})
    g_anc = pd.DataFrame({"sample_id": samples, "P0": 0.5, "P1": 0.5, "chrom": "1"})
    admix = np.zeros((3, len(samples), 2), dtype=np.int8)
    admix[..., 0] = [[2] * len(samples), [1] * len(samples), [0] * len(samples)]
    admix[..., 1] = 2 - admix[..., 0]
    admix[1, 0, :] = -1                              # one missing call on the second segment
    ds = from_legacy(loci, g_anc, admix)
    # segment ends: 100-799 and 800-.. are implied only for msp; here segment_end == pos,
    # so stepwise matches exact positions only and nearest with tolerance is the useful mode
    panel = hio.LocalAncestryPanel(ds)
    gen = InputGeneratorCisWithHaps(
        toy["genotype_df"], var_df, toy["phenotype_df"], toy["phenotype_pos_df"],
        haplotypes=panel, window=200, locus_method="nearest", nearest_tolerance=100,
    )
    # SNPs farther than 100 bp from every locus on their chromosome are dropped
    loci_by_chrom = {"1": np.array([100, 800]), "2": np.array([150])}
    dist = np.array([np.abs(loci_by_chrom[str(c)] - p).min() for c, p in zip(var_df["chrom"], var_df["pos"])])
    assert gen.n_unmapped_variants == int((dist > 100).sum()) > 0
    assert (~gen._vmask).sum() == gen.n_unmapped_variants
    for p, G, v_idx, H, pid in gen.generate_data():
        assert H.shape[0] == len(v_idx) == G.shape[0]
        assert not np.isnan(H).any()                 # the -1 became NaN and was interpolated


def test_generator_panel_mode_preload_to_torch(toy_hap_data_2anc):
    toy = toy_hap_data_2anc
    panel, admix = _panel_for_toy(toy)
    gen = InputGeneratorCisWithHaps(
        toy["genotype_df"], toy["variant_df"], toy["phenotype_df"], toy["phenotype_pos_df"],
        haplotypes=panel, window=200, preload_to_torch=True, torch_device=torch.device("cpu"),
    )
    for p, G, v_idx, H, pid in gen.generate_data():
        assert isinstance(H, torch.Tensor) and H.dtype == torch.float32 and H.device.type == "cpu"
        np.testing.assert_array_equal(H.numpy()[:, :, 0], admix[v_idx, :, 0])
    assert isinstance(gen._empty_h_block(), torch.Tensor)


def test_generator_panel_mode_rejects_sample_mismatch(toy_hap_data_2anc):
    toy = toy_hap_data_2anc
    panel, _ = _panel_for_toy(toy)
    geno = toy["genotype_df"].iloc[:, ::-1]          # reversed sample order
    with pytest.raises(ValueError, match="select_samples"):
        InputGeneratorCisWithHaps(geno, toy["variant_df"], toy["phenotype_df"].loc[:, geno.columns],
                                  toy["phenotype_pos_df"], haplotypes=panel, window=200)


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

    # toy_hap_data_2anc contains three non-constant phenotypes (geneA/B/C)
    # so we expect one batch per phenotype, matching InputGeneratorCis behavior.
    assert len(batches) == 3

    expected_pids = ["geneA", "geneB", "geneC"]

    for (p, G, v_idx, H, pid), expected_pid in zip(batches, expected_pids):
        # pid should be the phenotype ID from toy data
        assert pid == expected_pid

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


def test_preload_to_torch_returns_tensor_blocks(toy_hap_data_2anc):
    """When preloading, batches should yield torch tensors on the requested device."""
    geno_df = toy_hap_data_2anc["genotype_df"]
    var_df = toy_hap_data_2anc["variant_df"]
    pheno_df = toy_hap_data_2anc["phenotype_df"]
    pheno_pos_df = toy_hap_data_2anc["phenotype_pos_df"]
    hap = toy_hap_data_2anc["haplotypes"]
    loci_df = toy_hap_data_2anc["loci_df"]

    gen = InputGeneratorCisWithHaps(
        geno_df,
        var_df,
        pheno_df,
        pheno_pos_df,
        haplotypes=hap,
        loci_df=loci_df,
        on_the_fly_impute=True,
        window=200,
        preload_to_torch=True,
        torch_device=torch.device("cpu"),
    )

    batches = list(gen.generate_data())
    assert batches, "Generator should yield at least one batch"
    for batch in batches:
        assert len(batch) == 5
        H_block = batch[3]
        assert isinstance(H_block, torch.Tensor)
        assert H_block.device.type == "cpu"
        assert H_block.dtype == torch.float32

    empty = gen._empty_h_block()
    assert isinstance(empty, torch.Tensor)
    assert empty.device.type == "cpu"
