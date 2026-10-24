
import numpy as np
import pandas as pd
import pytest

from localqtl.freq import (
    FrequencyTable,
    read_frequency_table,
    resolve_ancestry_map,
    normalize_chrom,
    ensure_variant_keys,
    GRCH38_REFSEQ_TO_CHROM,
)


def _table():
    df = pd.DataFrame({
        "chrom": ["chr1", "1", "NC_000001.11", "chr1", "NC_000001.10"],
        "pos": [100, 110, 120, 120, 999],
        "ref": ["A", "C", "G", "G", "A"],
        "alt": ["G", "T", "A", "T", "C"],
        "af_afr": [0.30, 0.60, 0.25, 0.02, 0.5],
        "ac_afr": [30, 60, 25, 2, 5],
        "an_afr": [100, 100, 100, 100, 10],
        "af_nfe": [0.05, 0.40, 0.15, 0.0, 0.5],
        "ac_nfe": [5, 40, 15, 0, 5],
        "an_nfe": [100, 100, 100, 100, 10],
    })
    return FrequencyTable(df, source="unit")


def test_normalize_chrom_styles():
    assert normalize_chrom("7") == "chr7"
    assert normalize_chrom("chr7") == "chr7"
    assert normalize_chrom("MT") == "chrM"
    assert normalize_chrom("NC_000007.14") == "chr7"
    assert normalize_chrom("NC_000007.12") is None          # not GRCh38
    assert normalize_chrom("chrX", style="plain") == "X"
    assert len(GRCH38_REFSEQ_TO_CHROM) == 25


def test_table_normalises_and_drops_non_grch38_rows():
    t = _table()
    assert len(t) == 4                                      # NC_000001.10 row dropped
    assert set(t.df["chrom"]) == {"chr1"}
    assert t.groups == ["afr", "nfe"]
    with pytest.raises(ValueError, match="GRCh38"):
        FrequencyTable(t.df, source="x", build="GRCh37")
    with pytest.raises(ValueError, match="af_"):
        FrequencyTable(t.df[["chrom", "pos", "ref", "alt"]], source="x")
    with pytest.raises(ValueError, match="Unknown frequency group"):
        t.validate_groups(["eas"])
    sub = t.select_groups(["nfe"])
    assert sub.groups == ["nfe"] and "af_afr" not in sub.df.columns
    ren = t.rename_groups({"afr": "african"})
    assert "af_african" in ren.df.columns


def test_lookup_exact_flipped_ambiguous_missing(query_variants):
    t = _table()
    out = t.lookup(query_variants)
    assert list(out.index) == list(query_variants.index)
    # exact
    assert out.loc["v100", "match_type"] == "exact"
    np.testing.assert_allclose(out.loc["v100", "af_afr"], 0.30)
    assert bool(out.loc["v100", "strand_ambiguous"]) is False
    # flipped: table C/T, query T/C -> af = 1 - 0.60, ac = an - ac
    assert out.loc["v110", "match_type"] == "flipped"
    assert bool(out.loc["v110", "allele_flipped"])
    np.testing.assert_allclose(out.loc["v110", "af_afr"], 0.40)
    np.testing.assert_allclose(out.loc["v110", "ac_afr"], 40)
    # multi-allelic position: query G/T matches the second ALT row
    assert out.loc["v120", "match_type"] == "exact"
    np.testing.assert_allclose(out.loc["v120", "af_afr"], 0.02)
    # absent from table, strand ambiguous A/T
    assert out.loc["v140", "match_type"] == "none"
    assert bool(out.loc["v140", "missing_in_reference"])
    assert bool(out.loc["v140", "strand_ambiguous"])
    assert np.isnan(out.loc["v140", "af_afr"])
    # chrom given as "1" and pos 500000 absent
    assert out.loc["v5e5", "match_type"] == "none"
    # rows preserved even when the index is shuffled
    shuffled = query_variants.iloc[::-1]
    out2 = t.lookup(shuffled)
    assert list(out2.index) == list(shuffled.index)
    np.testing.assert_allclose(out2.loc["v110", "af_nfe"], 0.60)


def test_lookup_position_only_and_missing_alleles():
    t = _table()
    vdf = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "pos": [100, 120, 110]},
                       index=["a", "b", "c"])
    with pytest.raises(ValueError, match="ref/alt"):
        t.lookup(vdf)
    out = t.lookup(vdf, position_only=True)
    assert out.loc["a", "match_type"] == "position_only"
    assert out.loc["b", "match_type"] == "none"              # two table rows at pos 120 -> ambiguous
    np.testing.assert_allclose(out.loc["c", "af_afr"], 0.60)  # orientation NOT adjusted
    # IDs shaped like chr_pos_ref_alt are parsed when alleles are absent
    vdf2 = pd.DataFrame({"chrom": ["chr1"], "pos": [110]}, index=["chr1_110_T_C_b38"])
    keys = ensure_variant_keys(vdf2)
    assert keys.loc["chr1_110_T_C_b38", "alt"] == "C"
    out = t.lookup(vdf2)
    assert out.iloc[0]["match_type"] == "flipped"


def test_parquet_and_tsv_round_trip(tmp_path):
    t = _table()
    t.metadata["note"] = "hello"
    p = t.write(str(tmp_path / "freq.parquet"))
    back = read_frequency_table(p)
    assert back.source == "unit" and back.metadata["note"] == "hello"
    pd.testing.assert_frame_equal(back.df, t.df, check_dtype=False)
    p2 = t.write(str(tmp_path / "freq.tsv"))
    back2 = read_frequency_table(p2)
    assert back2.source == "unit"
    assert len(back2) == len(t)


def test_resolve_ancestry_map_requires_explicit_mapping():
    t = _table()
    with pytest.raises(ValueError, match="ancestry_map is required"):
        resolve_ancestry_map({}, ["AFR", "EUR"], t)
    with pytest.raises(ValueError, match="not panel ancestries"):
        resolve_ancestry_map({"AFRICAN": "afr"}, ["AFR", "EUR"], t)
    with pytest.raises(ValueError, match="Unknown frequency group"):
        resolve_ancestry_map({"AFR": "amr"}, ["AFR", "EUR"], t)
    assert resolve_ancestry_map({"AFR": "afr", "EUR": "nfe"}, ["AFR", "EUR"], t) == {"AFR": "afr", "EUR": "nfe"}
