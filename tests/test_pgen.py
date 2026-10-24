import numpy as np
import pandas as pd
import pytest

import sys
sys.path.append("src")
import localqtl.pgen as pgen

def test_impute_mean_vector():
    arr = np.array([1, -9, 2], dtype=np.int32)
    out = pgen._impute_mean(arr.copy(), missing_code=-9)
    assert out[1] == (1+2)/2

def test_impute_mean_matrix():
    arr = np.array([[1, -9, 2],
                    [0, 1, 2]], dtype=np.int32)
    out = pgen._impute_mean(arr.copy(), missing_code=-9)
    assert not (out == -9).any()

def test_read_pvar_and_psam(tmp_path):
    pvar = tmp_path / "toy.pvar"
    psam = tmp_path / "toy.psam"
    pvar.write_text("1\t100\trs1\tA\tG\t.\t.\t.")
    psam.write_text("IID\ns1\ns2\n")
    
    pvar_df = pgen.read_pvar(str(pvar))
    psam_df = pgen.read_psam(str(psam))
    assert "chrom" in pvar_df
    assert psam_df.index.tolist() == ["s1","s2"]


def test_generate_paired_chunks(tmp_path):
    from tests.conftest import make_cohort_with_chroms, write_toy_pgen
    d = make_cohort_with_chroms(n=20, m_per_chrom=(8, 6), n_pheno_per_chrom=(3, 2))
    prefix = write_toy_pgen(str(tmp_path / "toy"), d["genotype_df"], d["variant_df"])
    pgr = pgen.PgenReader(prefix, select_samples=d["samples"])
    assert pgr.variant_df.loc["v1_0", "alt"] == "G"
    assert pgen.get_chunk_size(1, 500) == 1024 ** 3 // 500
    chunks = list(pgen.generate_paired_chunks(pgr, d["phenotype_df"], d["phenotype_pos_df"], chunk_size="chr",
                                              window=10_000, verbose=False))
    assert len(chunks) == 2
    gt, var, p, ppos, ci = chunks[0]
    assert ci == 0 and list(p.index) == ["gene1_0", "gene1_1", "gene1_2"] and (var["chrom"] == "chr1").all()
    assert gt.shape == (8, 20) and list(gt.columns) == d["samples"]
    np.testing.assert_allclose(gt.to_numpy(np.float32), d["genotype_df"].loc[var.index].to_numpy())
    # variant-count chunks: every cis window fits, phenotypes partition the input
    chunks2 = list(pgen.generate_paired_chunks(pgr, d["phenotype_df"], d["phenotype_pos_df"], chunk_size=8,
                                               window=1_000, verbose=False))
    assert sum(len(c[2]) for c in chunks2) == 5
    for gt, var, p, ppos, _ in chunks2:
        assert gt.shape[0] <= 8 and list(var.index) == list(gt.index)
    with pytest.raises(ValueError, match="chunk size"):
        list(pgen.generate_paired_chunks(pgr, d["phenotype_df"], d["phenotype_pos_df"], chunk_size=2,
                                         window=10_000, verbose=False))
