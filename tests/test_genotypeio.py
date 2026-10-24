import numpy as np
import pandas as pd
import pytest

from src.localqtl.genotypeio import (
    _impute_mean,
    get_cis_ranges,
    InputGeneratorCis,
)


############################
# _impute_mean tests
############################

def test_impute_mean_basic():
    """
    Rows with -9 should get imputed to that row's mean of non-missing.
    Also: dtype must be float32/float64.
    """
    g = np.array([
        [1.0, -9.0, 2.0],
        [2.0,  2.0, 2.0],
    ], dtype=np.float32)

    _impute_mean(g, missing=-9, verbose=False)

    # Row 0 mean of [1,2] = 1.5, so missing becomes 1.5
    assert np.isclose(g[0, 1], 1.5)
    # Row 1 had no missing, unchanged
    assert np.allclose(g[1], [2.0, 2.0, 2.0])


def test_impute_mean_bad_dtype_raises():
    """
    _impute_mean should complain if dtype isn't float32/float64.
    """
    g_bad = np.array([[0, -9, 1]], dtype=np.int8)
    with pytest.raises(ValueError):
        _impute_mean(g_bad, missing=-9)


############################
# get_cis_ranges tests
############################

def test_get_cis_ranges_window100(toy_data):
    """
    Make sure cis windows are computed correctly for each phenotype, and
    phenotypes with no variants in cis get dropped_ids.
    """
    genotype_df       = toy_data["genotype_df"]
    variant_df        = toy_data["variant_df"]
    phenotype_pos_df  = toy_data["phenotype_pos_df"]

    # Build chr_variant_dfs the same way InputGeneratorCis._calculate_cis_ranges does
    variant_df_tmp = variant_df.copy()
    variant_df_tmp["index"] = np.arange(variant_df_tmp.shape[0])
    chr_variant_dfs = {
        chrom: g[["pos", "index"]]
        for chrom, g in variant_df_tmp.groupby("chrom")
    }

    cis_ranges, drop_ids = get_cis_ranges(
        phenotype_pos_df=phenotype_pos_df.copy(),
        chr_variant_dfs=chr_variant_dfs,
        window=100,
        verbose=False
    )

    # geneA at chr1:180 with window=100 → [80,280] hits chr1 pos 100,200
    # variant_df_tmp rows: v1_chr1_100 -> row0, v2_chr1_200 -> row1
    assert np.allclose(cis_ranges["geneA"], [0, 1])

    # geneB at chr1:850-900 with window=100 → [750,1000] hits chr1 pos 800,900
    # variant_df_tmp rows: v4_chr1_800 -> row3, v5_chr1_900 -> row4
    assert np.allclose(cis_ranges["geneB"], [3, 4])

    # geneC at chr2:260 with window=100 → [160,360] hits chr2 pos 250 only
    # chr2 variants begin at row5 in the combined table (150,250,400)
    # Window excludes 150 (<160) and 400 (>360), so we only get row6 == v2_chr2_250
    assert np.allclose(cis_ranges["geneC"], [6, 6])

    # geneConst at chr1:500 with window=100 → [400,600]
    # there is no variant between 400 and 600 in chr1 (we jump from 300 to 800),
    # so geneConst should be dropped
    assert "geneConst" in drop_ids
    assert "geneConst" not in cis_ranges


############################
# InputGeneratorCis tests
############################

def test_input_generator_init_filters(toy_data):
    """
    InputGeneratorCis should:
    - keep phenotypes on chromosomes that have genotypes
    - drop any constant phenotype rows
    - drop phenotypes with no cis variants
    and populate internal fields correctly.
    """

    ig = InputGeneratorCis(
        genotype_df      = toy_data["genotype_df"],
        variant_df       = toy_data["variant_df"],
        phenotype_df     = toy_data["phenotype_df"],
        phenotype_pos_df = toy_data["phenotype_pos_df"],
        group_s          = None,
        window           = 100,
    )

    # geneConst should be removed because it's constant AND/OR has no window
    assert list(ig.phenotype_df.index) == ["geneA", "geneB", "geneC"]

    # n_phenotypes should match remaining rows
    assert ig.n_phenotypes == 3

    # n_samples should match #columns in phenotype/genotype (4 samples)
    assert ig.n_samples == 4

    # phenotype_start/phenotype_end dicts should exist and match positions
    assert ig.phenotype_start["geneA"] == 180
    assert ig.phenotype_end["geneA"]   == 180

    # captured chromosomes should be ["1","2"] in some order
    assert sorted(ig.chrs) == ["1", "2"]

    # cis_ranges keys should match the kept phenotypes
    assert set(ig.cis_ranges.keys()) == {"geneA", "geneB", "geneC"}


def test_input_generator_batches(toy_data):
    """
    The generator should yield batches of:
        p        (phenotype vector)
        G        (cis genotype slice)
        v_idx    (global row indices into genotype_df / variant_df)
        pid      (phenotype id)
    in the same order as ig.phenotype_df.index after filtering.
    """

    ig = InputGeneratorCis(
        genotype_df      = toy_data["genotype_df"],
        variant_df       = toy_data["variant_df"],
        phenotype_df     = toy_data["phenotype_df"],
        phenotype_pos_df = toy_data["phenotype_pos_df"],
        group_s          = None,
        window           = 100,
    )

    # generate_data() returns a BackgroundGenerator iterator
    it = ig.generate_data(chrom=None, verbose=False)

    # ---- first batch: geneA ----
    p, G, v_idx, pid = next(it)
    assert pid == "geneA"

    # p is shape (n_samples,)
    assert p.shape == (4,)
    # geneA: window should include variant rows 0 and 1 (chr1:100,200)
    assert np.allclose(v_idx, [0,1])
    # G should match those genotype rows
    np.testing.assert_array_equal(
        G,
        toy_data["genotype_df"].values[0:2, :]  # rows 0 and 1
    )

    # ---- second batch: geneB ----
    p2, G2, v_idx2, pid2 = next(it)
    assert pid2 == "geneB"
    # geneB: window hits rows 3 and 4 (chr1:800,900)
    assert np.allclose(v_idx2, [3,4])
    assert G2.shape == (2, 4)

    # ---- third batch: geneC ----
    p3, G3, v_idx3, pid3 = next(it)
    assert pid3 == "geneC"
    # geneC: window hits only row 6 (chr2:250)
    assert np.allclose(v_idx3, [6])
    assert G3.shape == (1, 4)

    # ---- iterator should now be exhausted ----
    with pytest.raises(StopIteration):
        next(it)


############################
# VCF helpers (tabix)
############################

def _write_vcf(tmp_path):
    pysam = pytest.importorskip("pysam")
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">',
        "##contig=<ID=chr1,length=100000>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3",
        "chr1\t100\tchr1_100_A_G_b38\tA\tG\t.\tPASS\t.\tGT:DS\t0/0:0.1\t0|1:1.0\t1/1:1.9",
        "chr1\t200\tchr1_200_C_T_b38\tC\tT\t.\tPASS\t.\tGT:DS\t0/1:0.9\t./.:.\t0/0:0.0",
        "chr1\t300\tchr1_300_G_A_b38\tG\tA\t.\tPASS\t.\tGT:DS\t1/1:2.0\t1/1:2.0\t0/1:1.1",
    ]
    plain = tmp_path / "toy.vcf"
    plain.write_text("\n".join(lines) + "\n")
    gz = tmp_path / "toy.vcf.gz"
    pysam.tabix_compress(str(plain), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="vcf", force=True)
    return str(gz)


def test_get_vcf_region_and_variants(tmp_path):
    from src.localqtl.genotypeio import get_sample_ids, get_vcf_region, get_vcf_variants
    vcf = _write_vcf(tmp_path)
    assert get_sample_ids(vcf) == ["S1", "S2", "S3"]
    df, pos_s = get_vcf_region("chr1:150-350", vcf)
    assert list(df.index) == ["chr1_200_C_T_b38", "chr1_300_G_A_b38"]
    assert list(pos_s) == [200, 300]
    np.testing.assert_allclose(df.loc["chr1_200_C_T_b38"], [1.0, 0.5, 0.0])  # missing imputed to the mean
    df2, _ = get_vcf_region("chr1", vcf, field="DS", select_samples=["S3", "S1"], impute_missing=False)
    assert list(df2.columns) == ["S3", "S1"] and df2.shape[0] == 3
    assert np.isnan(df2.loc["chr1_200_C_T_b38", "S1"]) is np.False_ and df2.loc["chr1_300_G_A_b38", "S3"] == pytest.approx(1.1)
    # contig name without the chr prefix is resolved
    df3, _ = get_vcf_region("1:100-100", vcf)
    assert list(df3.index) == ["chr1_100_A_G_b38"]
    g = get_vcf_variants(["chr1_300_G_A_b38", "chr1_100_A_G_b38", "chr1_200_C_T_b38"], vcf)
    # the variant with a missing call is dropped (tensorQTL behaviour)
    assert set(g.index) == {"chr1_100_A_G_b38", "chr1_300_G_A_b38"}
    np.testing.assert_allclose(g.loc["chr1_100_A_G_b38"], [0, 1, 2])
    g2 = get_vcf_variants({"chr1_200_C_T_b38": "chr1:200"}, vcf, field="DS")
    np.testing.assert_allclose(g2.loc["chr1_200_C_T_b38"], [0.9, np.nan, 0.0])
    with pytest.raises(ValueError, match="parse"):
        get_vcf_variants(["rs123"], vcf)
