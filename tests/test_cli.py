import gzip
import os

import numpy as np
import pandas as pd
import pytest

from localqtl.cli import main
from tests.conftest import make_cohort_with_chroms, write_phenotype_bed, write_toy_pgen


@pytest.fixture
def cli_inputs(tmp_path):
    d = make_cohort_with_chroms(seed=7, n=60, m_per_chrom=(8, 6), n_pheno_per_chrom=(2, 1), effect=1.0)
    geno = write_toy_pgen(str(tmp_path / "geno"), d["genotype_df"], d["variant_df"])
    bed = write_phenotype_bed(str(tmp_path / "pheno.bed.gz"), d["phenotype_df"], d["phenotype_pos_df"])
    cov = tmp_path / "covariates.txt"
    d["covariates_df"].T.to_csv(cov, sep="\t")
    inter = tmp_path / "interaction.txt"
    pd.DataFrame({"sex": d["rng"].normal(size=60)}, index=d["samples"]).to_csv(inter, sep="\t")
    return dict(d, geno=geno, bed=bed, cov=str(cov), inter=str(inter), out=str(tmp_path / "out"))


def _run(args):
    assert main(args) == 0


def test_cli_cis_pipeline(cli_inputs):
    c = cli_inputs
    common = [c["geno"], c["bed"], "toy", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
              "--window", "10000", "--seed", "1"]
    _run(common + ["--mode", "cis", "--permutations", "50"])
    cis_file = os.path.join(c["out"], "toy.cis_qtl.txt.gz")
    cis_df = pd.read_csv(cis_file, sep="\t")
    assert {"phenotype_id", "variant_id", "pval_beta", "qval", "true_df", "pval_true_df"} <= set(cis_df.columns)
    assert "true_dof" not in cis_df.columns and len(cis_df) == 3
    assert os.path.exists(os.path.join(c["out"], "toy.localQTL.cis.log"))

    _run(common + ["--mode", "cis_nominal", "--cis_output", cis_file, "--fdr", "0.5"])
    pairs = pd.read_parquet(os.path.join(c["out"], "toy.cis_qtl_pairs.chr1.parquet"))
    assert {"phenotype_id", "variant_id", "pval_nominal", "slope", "slope_se", "af"} <= set(pairs.columns)
    assert os.path.exists(os.path.join(c["out"], "toy.cis_qtl_pairs.chr2.parquet"))
    signif = pd.read_parquet(os.path.join(c["out"], "toy.cis_qtl.signif_pairs.parquet"))
    assert "pval_nominal_threshold_pheno" in signif.columns

    _run(common + ["--mode", "cis_independent", "--cis_output", cis_file, "--permutations", "30", "--fdr", "0.5"])
    ind = pd.read_csv(os.path.join(c["out"], "toy.cis_independent_qtl.txt.gz"), sep="\t")
    assert {"phenotype_id", "variant_id", "rank"} <= set(ind.columns) and len(ind) >= 1

    _run(common + ["--mode", "cis_susie", "--cis_output", cis_file, "--fdr", "0.5", "--max_effects", "3"])
    assert os.path.exists(os.path.join(c["out"], "toy.SuSiE_summary.parquet"))
    assert os.path.exists(os.path.join(c["out"], "toy.SuSiE.pickle"))


def test_cli_cis_nominal_interaction_and_best_only(cli_inputs):
    c = cli_inputs
    common = [c["geno"], c["bed"], "toy", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
              "--window", "10000", "--interaction", c["inter"], "--maf_threshold_interaction", "0"]
    _run(common + ["--mode", "cis_nominal"])
    top = pd.read_csv(os.path.join(c["out"], "toy.cis_qtl_top_assoc.txt.gz"), sep="\t", index_col=0)
    assert {"pval_gi", "b_gi", "b_gi_se", "tests_emt", "pval_emt", "pval_adj_bh"} <= set(top.columns)
    assert len(top) == 3
    pairs = pd.read_parquet(os.path.join(c["out"], "toy.cis_qtl_pairs.chr1.parquet"))
    assert "b_gi" in pairs.columns
    out2 = c["out"] + "_best"
    _run(common[:-6] + ["-o", out2, "--device", "cpu", "--window", "10000", "--interaction", c["inter"],
                        "--mode", "cis_nominal", "--best_only"])
    assert os.path.exists(os.path.join(out2, "toy.cis_qtl_top_assoc.txt.gz"))
    assert not [f for f in os.listdir(out2) if f.endswith(".parquet")]
    with pytest.raises(ValueError, match="Interactions"):
        main(common + ["--mode", "cis"])


def test_cli_chunked_cis_nominal(cli_inputs):
    c = cli_inputs
    _run([c["geno"], c["bed"], "toy", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--window", "10000", "--mode", "cis_nominal", "--chunk_size", "chr"])
    files = sorted(f for f in os.listdir(c["out"]) if f.endswith(".parquet"))
    assert files == ["toy.cis_qtl_pairs.chr1.parquet", "toy.cis_qtl_pairs.chr2.parquet"]
    _run([c["geno"], c["bed"], "toy2", "-o", c["out"], "--device", "cpu", "--window", "10000",
          "--mode", "cis", "--permutations", "20", "--chunk_size", "8"])
    assert len(pd.read_csv(os.path.join(c["out"], "toy2.cis_qtl.txt.gz"), sep="\t")) == 3


def test_cli_trans_sparse_dense_and_text(cli_inputs):
    c = cli_inputs
    base = [c["geno"], c["bed"], "toy", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
            "--mode", "trans", "--maf_threshold", "0"]
    _run(base + ["--pval_threshold", "1", "--return_r2"])
    pairs = pd.read_parquet(os.path.join(c["out"], "toy.trans_qtl_pairs.parquet"))
    assert list(pairs.columns) == ["variant_id", "phenotype_id", "pval", "b", "b_se", "r2", "af"]
    # cis pairs (within 5 Mb on the same chromosome) are removed: only cross-chromosome pairs remain
    assert len(pairs) == 8 * 1 + 6 * 2
    _run(base + ["--pval_threshold", "1", "--output_text"])
    with gzip.open(os.path.join(c["out"], "toy.trans_qtl_pairs.txt.gz"), "rt") as f:
        assert f.readline().startswith("variant_id\tphenotype_id\tpval")
    _run(base + ["--return_dense"])
    for suffix in ("pval", "beta", "beta_se", "af"):
        assert os.path.exists(os.path.join(c["out"], f"toy.trans_qtl_{suffix}.parquet"))
    pval_df = pd.read_parquet(os.path.join(c["out"], "toy.trans_qtl_pval.parquet"))
    assert pval_df.shape == (14, 3)
    # chunked trans + interaction
    _run(base + ["--pval_threshold", "1", "--chunk_size", "5", "--interaction", c["inter"]])
    pairs = pd.read_parquet(os.path.join(c["out"], "toy.trans_qtl_pairs.parquet"))
    assert {"pval_g", "pval_i", "pval_gi", "b_gi", "b_gi_se"} <= set(pairs.columns)
    # phenotypes as a plain table (no positions): no cis filtering
    tbl = os.path.join(c["out"], "pheno.txt")
    c["phenotype_df"].to_csv(tbl, sep="\t")
    _run([c["geno"], tbl, "toy3", "-o", c["out"], "--device", "cpu", "--mode", "trans", "--maf_threshold", "0",
          "--pval_threshold", "1"])
    assert len(pd.read_parquet(os.path.join(c["out"], "toy3.trans_qtl_pairs.parquet"))) == 14 * 3


def test_cli_trans_susie_and_version(cli_inputs, capsys):
    c = cli_inputs
    loci = os.path.join(c["out"] + "_loci", "loci.tsv")
    os.makedirs(os.path.dirname(loci), exist_ok=True)
    pd.DataFrame({"phenotype_id": ["gene1_0"], "chr": ["chr1"], "position": [1200]}).to_csv(loci, sep="\t", index=False)
    _run([c["geno"], c["bed"], "toy", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--mode", "trans_susie", "--susie_loci", loci, "--window", "10000", "--maf_threshold", "0"])
    assert os.path.exists(os.path.join(c["out"], "toy.SuSiE_summary.parquet"))
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "localqtl" in capsys.readouterr().out


def test_cli_local_ancestry_cis(cli_inputs, tmp_path, monkeypatch):
    """--haplotypes: main-effect LA covariates in cis and trans modes (rfmix_reader cache when writable)."""
    pytest.importorskip("rfmix_reader")
    from rfmix_reader import from_legacy
    from localqtl import cli
    from localqtl.haplotypeio import LocalAncestryPanel
    c = cli_inputs
    var_df = c["variant_df"]
    loci = pd.DataFrame({"chromosome": var_df["chrom"].astype(str).values, "physical_position": var_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": c["samples"], "P0": 0.5, "P1": 0.5, "chrom": "1"})
    codes = c["rng"].integers(0, 2, size=(len(var_df), len(c["samples"]), 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(2)], axis=-1).astype(np.int8)
    ds = from_legacy(loci, g_anc, admix)
    cache = str(tmp_path / "la_cache")
    os.makedirs(cache)
    try:
        for chrom in ds.la.chromosomes:
            ds.la.sel_chrom(chrom).to_zarr(os.path.join(cache, f"{chrom}.zarr"))
        from rfmix_reader import open_local_ancestry
        open_local_ancestry(cache)
        hap_args = ["--haplotypes", cache, "--la_format", "cache"]
    except Exception:  # cache layout not reproducible here: exercise the CLI wiring with the in-memory panel
        monkeypatch.setattr(cli, "load_haplotypes",
                            lambda args, sample_ids, logger: LocalAncestryPanel(ds, select_samples=list(sample_ids)))
        hap_args = ["--haplotypes", cache]
    _run([c["geno"], c["bed"], "la", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--window", "10000", "--mode", "cis", "--permutations", "20"] + hap_args)
    cis = pd.read_csv(os.path.join(c["out"], "la.cis_qtl.txt.gz"), sep="\t")
    assert len(cis) == 3
    _run([c["geno"], c["bed"], "la", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--mode", "trans", "--maf_threshold", "0", "--pval_threshold", "1"] + hap_args)
    assert len(pd.read_parquet(os.path.join(c["out"], "la.trans_qtl_pairs.parquet"))) == 8 * 1 + 6 * 2
    # dosage model through the CLI
    _run([c["geno"], c["bed"], "la", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--window", "10000", "--mode", "cis_nominal", "--ancestry_model", "dosage", "--min_ancestry_ac", "1"] + hap_args)
    pairs = pd.read_parquet(os.path.join(c["out"], "la.cis_qtl_pairs.chr1.parquet"))
    assert {"slope_anc0", "slope_anc1", "pval_joint", "af_anc0"} <= set(pairs.columns)


def test_cli_cis_dosage_and_paired_covariate(cli_inputs, tmp_path, monkeypatch):
    """--ancestry_model dosage in `cis` mode, and --paired_covariate in cis_independent/cis_susie."""
    pytest.importorskip("rfmix_reader")
    from rfmix_reader import from_legacy
    from localqtl import cli
    from localqtl.haplotypeio import LocalAncestryPanel
    c = cli_inputs
    var_df = c["variant_df"]
    loci = pd.DataFrame({"chromosome": var_df["chrom"].astype(str).values,
                         "physical_position": var_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": c["samples"], "P0": 0.5, "P1": 0.5, "chrom": "1"})
    codes = c["rng"].integers(0, 2, size=(len(var_df), len(c["samples"]), 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(2)], axis=-1).astype(np.int8)
    ds = from_legacy(loci, g_anc, admix)
    monkeypatch.setattr(cli, "load_haplotypes",
                        lambda args, sample_ids, logger: LocalAncestryPanel(ds, select_samples=list(sample_ids)))

    # `cis` mode with the dosage model: permutations calibrate the joint F
    _run([c["geno"], c["bed"], "dos", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
          "--window", "10000", "--mode", "cis", "--permutations", "30", "--haplotypes", "x",
          "--ancestry_model", "dosage", "--min_ancestry_ac", "1"])
    cis_df = pd.read_csv(os.path.join(c["out"], "dos.cis_qtl.txt.gz"), sep="\t")
    assert {"slope_anc0", "slope_anc1", "pval_joint", "af_anc0", "pval_perm", "pval_beta"} <= set(cis_df.columns)
    assert len(cis_df) == 3 and cis_df["pval_perm"].between(0, 1).all()

    # a phenotype-specific covariate now also reaches cis_independent and cis_susie
    paired = os.path.join(c["out"], "paired.txt")
    pd.DataFrame(c["rng"].normal(size=(1, len(c["samples"]))),
                 index=["gene1_0"], columns=c["samples"]).to_csv(paired, sep="\t")
    base = [c["geno"], c["bed"], "pc", "--covariates", c["cov"], "-o", c["out"], "--device", "cpu",
            "--window", "10000", "--paired_covariate", paired]
    _run(base + ["--mode", "cis", "--permutations", "30"])
    cis_file = os.path.join(c["out"], "pc.cis_qtl.txt.gz")
    _run(base + ["--mode", "cis_independent", "--cis_output", cis_file, "--permutations", "30", "--fdr", "0.5"])
    assert os.path.exists(os.path.join(c["out"], "pc.cis_independent_qtl.txt.gz"))
    _run(base + ["--mode", "cis_susie", "--cis_output", cis_file, "--fdr", "0.5", "--max_effects", "2"])
    assert os.path.exists(os.path.join(c["out"], "pc.SuSiE_summary.parquet"))

    with pytest.raises(ValueError, match="dosage is only available"):
        main([c["geno"], c["bed"], "x", "-o", c["out"], "--device", "cpu", "--mode", "cis_independent",
              "--cis_output", cis_file, "--haplotypes", "x", "--ancestry_model", "dosage"])
