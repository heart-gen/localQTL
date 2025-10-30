import pandas as pd

from src.localqtl.cis.postproc import _chrom_sort_key, get_significant_pairs
from src.localqtl.utils import SimpleLogger


def test_chrom_sort_key_handles_special_chromosomes():
    chroms = ["chrX", "chr2", "chrM", "chr10", "chrY", "chr1", "custom"]
    ordered = sorted(chroms, key=_chrom_sort_key)
    assert ordered[:4] == ["chr1", "chr2", "chr10", "chrX"]
    assert ordered[4:] == ["chrY", "chrM", "custom"]


def _write_nominal(tmp_path, chrom, rows):
    path = tmp_path / f"cis_nominal.{chrom}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_get_significant_pairs_filters_nominal_hits(tmp_path):
    res_df = pd.DataFrame({
        "phenotype_id": ["geneA", "geneB", "geneC"],
        "qval": [0.01, 0.2, 0.03],
        "pval_nominal_threshold": [0.05, 0.05, 0.01],
    })

    _write_nominal(tmp_path, "chr1", [
        {"phenotype_id": "geneA", "variant_id": "v1", "pval_nominal": 0.04},
        {"phenotype_id": "geneB", "variant_id": "v2", "pval_nominal": 0.01},
    ])
    _write_nominal(tmp_path, "chr2", [
        {"phenotype_id": "geneC", "variant_id": "v3", "pval_nominal": 0.005},
    ])

    result = get_significant_pairs(
        res_df=res_df,
        nominal_files=str(tmp_path / "cis_nominal.chr*.parquet"),
        fdr=0.05,
        logger=SimpleLogger(verbose=False),
    )

    assert set(result["phenotype_id"]) == {"geneA", "geneC"}
    assert (result["pval_nominal"] <= result["pval_nominal_threshold_pheno"]).all()


def test_get_significant_pairs_group_mode(tmp_path):
    res_df = pd.DataFrame({
        "group_id": ["g1", "g2"],
        "qval": [0.01, 0.2],
        "pval_nominal_threshold": [0.02, 0.5],
    })
    group_s = pd.Series({"geneA": "g1", "geneB": "g1", "geneC": "g2"}, name="phenotype_id")

    _write_nominal(tmp_path, "chr1", [
        {"phenotype_id": "geneA", "variant_id": "v1", "pval_nominal": 0.01},
        {"phenotype_id": "geneB", "variant_id": "v2", "pval_nominal": 0.5},
    ])

    result = get_significant_pairs(
        res_df=res_df,
        nominal_files={"chr1": tmp_path / "cis_nominal.chr1.parquet"},
        group_s=group_s,
        fdr=0.05,
        logger=SimpleLogger(verbose=False),
    )

    assert set(result["group_id"]) == {"g1"}
    assert (result["pval_nominal"] <= result["pval_nominal_threshold_group"]).all()
