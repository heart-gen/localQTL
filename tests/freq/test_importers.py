import numpy as np
import pandas as pd
import pytest

from localqtl.freq import (
    from_gnomad_vcf, inspect_gnomad_header, from_alfa_vcf, from_bcftools_query_tsv,
    regions_from_variant_df, regions_to_bed, regions_to_refseq_bed, ALFA_POPULATIONS,
)
from localqtl.freq import _vcf as vcfmod


def test_regions_from_variant_df_merges_clusters(query_variants):
    r = regions_from_variant_df(query_variants, gap=1000)
    assert list(r["chrom"]) == ["chr1", "chr1"]
    assert r.iloc[0]["start"] == 100 and r.iloc[0]["end"] == 140
    assert r.iloc[1]["start"] == 500000


def test_gnomad_importer_reads_groups_tracts_and_filters(gnomad_like_vcf, query_variants):
    pytest.importorskip("pysam")
    hdr = inspect_gnomad_header(gnomad_like_vcf)
    assert "AF_afr_tract" in set(hdr["field"])
    t = from_gnomad_vcf(gnomad_like_vcf, variant_df=query_variants, groups=("afr", "nfe", "eas"),
                        biallelic_only=False)
    # groups: global 'all', afr, nfe, auto-detected tract (sex-stratified XX excluded, grpmax excluded)
    assert set(t.groups) == {"all", "afr", "nfe", "lai_afr_tract"}
    assert t.metadata["tract_fields"]["lai_afr_tract"] == ["AF_afr_tract", "AC_afr_tract", "AN_afr_tract"]
    assert "info_af_grpmax" in t.df.columns and "grpmax" not in t.groups
    # non-PASS row (pos 130) dropped; multi-allelic expanded; pos 140 absent
    assert set(t.df["pos"]) == {100, 110, 120, 500000}
    assert (t.df["pos"] == 120).sum() == 2
    row = t.df.set_index(["pos", "alt"]).loc[(120, "T")]
    np.testing.assert_allclose(row["af_afr"], 0.02)
    np.testing.assert_allclose(row["an_afr"], 100)
    assert t.df.set_index("pos").loc[100, "rsid"] == "rs100"
    # biallelic_only drops the multi-allelic site; pass_only=False keeps pos 130
    t2 = from_gnomad_vcf(gnomad_like_vcf, variant_df=query_variants, pass_only=False, tract_fields=None)
    assert 130 in set(t2.df["pos"]) and 120 not in set(t2.df["pos"])
    assert not any(g.startswith("lai_") for g in t2.groups)
    # explicit tract list must exist
    with pytest.raises(ValueError, match="tract field"):
        from_gnomad_vcf(gnomad_like_vcf, variant_df=query_variants, tract_fields=["nope"])
    # lookup orientation through the importer output
    look = t.lookup(query_variants)
    assert look.loc["v110", "match_type"] == "flipped"
    np.testing.assert_allclose(look.loc["v110", "af_afr"], 0.40)


def test_alfa_importer_maps_accessions_and_populations(alfa_like_vcf, query_variants):
    pytest.importorskip("pysam")
    t = from_alfa_vcf(alfa_like_vcf, variant_df=query_variants, populations=("EUR", "AFR"))
    assert set(t.groups) == {"eur", "afr"}
    assert set(t.df["chrom"]) == {"chr1"}
    # pos 100 appears once (GRCh37 accession row never fetched / dropped)
    assert (t.df["pos"] == 100).sum() == 1
    row = t.df.set_index(["pos", "alt"])
    np.testing.assert_allclose(row.loc[(100, "G"), "af_eur"], 0.05)
    np.testing.assert_allclose(row.loc[(100, "G"), "af_afr"], 0.30)
    np.testing.assert_allclose(row.loc[(120, "T"), "ac_afr"], 8)
    assert np.isnan(row.loc[(130, "C"), "af_eur"])          # AN == 0 -> NaN
    np.testing.assert_allclose(row.loc[(130, "C"), "af_afr"], 0.01)
    assert t.metadata["population_names"]["AFR"].startswith("African")
    with pytest.raises(ValueError, match="Unknown ALFA population"):
        from_alfa_vcf(alfa_like_vcf, variant_df=query_variants, populations=("XYZ",))


def test_missing_pysam_gives_install_hint(monkeypatch, query_variants):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pysam":
            raise ImportError("no pysam")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(vcfmod, "_PYSAM_AVAILABLE", None)
    with pytest.raises(ImportError, match=r'localqtl\[annotate\]'):
        from_gnomad_vcf("nothing.vcf.bgz", variant_df=query_variants)
    assert vcfmod.pysam_available() is False


def test_bcftools_tsv_reader_gnomad_and_alfa_layouts(tmp_path):
    g = tmp_path / "gnomad.tsv"
    g.write_text(
        "# [1]CHROM\t[2]POS\t[3]ID\t[4]REF\t[5]ALT\t[6]INFO/AF_afr\t[7]INFO/AC_afr\t[8]INFO/AN_afr\t[9]INFO/AF_nfe\t[10]INFO/AC_nfe\t[11]INFO/AN_nfe\n"
        "chr1\t100\trs100\tA\tG\t0.3\t30\t100\t0.05\t5\t100\n"
        "chr1\t120\t.\tG\tA,T\t0.25,0.02\t25,2\t100\t0.15,.\t15,.\t100\n"
    )
    t = from_bcftools_query_tsv(str(g))
    assert set(t.groups) == {"afr", "nfe"}
    assert len(t) == 3
    r = t.df.set_index(["pos", "alt"])
    np.testing.assert_allclose(r.loc[(120, "T"), "af_afr"], 0.02)
    assert np.isnan(r.loc[(120, "T"), "af_nfe"])
    assert r.loc[(100, "G"), "rsid"] == "rs100"

    a = tmp_path / "alfa.tsv"
    a.write_text(
        "# [1]CHROM\t[2]POS\t[3]ID\t[4]REF\t[5]ALT\t[6]SAMN10492703:AN\t[7]SAMN10492703:AC\t[8]SAMN10492695:AN\t[9]SAMN10492695:AC\n"
        "NC_000001.11\t100\trs100\tA\tG\t400\t120\t1000\t50\n"
        "NC_000001.10\t100\trs100\tA\tG\t10\t1\t10\t1\n"
    )
    t2 = from_bcftools_query_tsv(str(a), source="alfa")
    assert set(t2.groups) == {"afr", "eur"}
    assert len(t2) == 1                                       # GRCh37 accession dropped
    np.testing.assert_allclose(t2.df.iloc[0]["af_afr"], 0.30)  # af derived from ac/an
    assert ALFA_POPULATIONS["SAMN10492703"] == "AFR"

    with pytest.raises(ValueError, match="header"):
        bad = tmp_path / "bad.tsv"
        bad.write_text("chr1\t100\tA\tG\t0.1\n")
        from_bcftools_query_tsv(str(bad))


def test_region_bed_writers(tmp_path):
    regions = pd.DataFrame({"chrom": ["chr1", "7"], "start": [100, 5000], "end": [200, 6000]})
    p = regions_to_bed(regions, str(tmp_path / "r.bed"))
    lines = open(p).read().splitlines()
    assert lines[0] == "chr1\t99\t200" and lines[1] == "chr7\t4999\t6000"
    p2 = regions_to_refseq_bed(regions, str(tmp_path / "r.refseq.bed"))
    lines = open(p2).read().splitlines()
    assert lines[0].startswith("NC_000001.11\t99") and lines[1].startswith("NC_000007.14\t4999")
