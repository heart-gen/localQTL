import numpy as np
import pandas as pd
import pytest


def _write_tabix_vcf(tmp_path, name, header_lines, records):
    """Write a bgzipped + tabix-indexed VCF; returns the .vcf.gz path."""
    pysam = pytest.importorskip("pysam")
    plain = tmp_path / f"{name}.vcf"
    plain.write_text("\n".join(header_lines + records) + "\n")
    gz = tmp_path / f"{name}.vcf.gz"
    pysam.tabix_compress(str(plain), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="vcf", force=True)
    return str(gz)


@pytest.fixture
def gnomad_like_vcf(tmp_path):
    header = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FILTER=<ID=AC0,Description="no high quality genotype">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="AF">',
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="AC">',
        '##INFO=<ID=AN,Number=1,Type=Integer,Description="AN">',
        '##INFO=<ID=AF_afr,Number=A,Type=Float,Description="">',
        '##INFO=<ID=AC_afr,Number=A,Type=Integer,Description="">',
        '##INFO=<ID=AN_afr,Number=1,Type=Integer,Description="">',
        '##INFO=<ID=AF_nfe,Number=A,Type=Float,Description="">',
        '##INFO=<ID=AC_nfe,Number=A,Type=Integer,Description="">',
        '##INFO=<ID=AN_nfe,Number=1,Type=Integer,Description="">',
        '##INFO=<ID=AF_afr_tract,Number=A,Type=Float,Description="LAI">',
        '##INFO=<ID=AC_afr_tract,Number=A,Type=Integer,Description="LAI">',
        '##INFO=<ID=AN_afr_tract,Number=1,Type=Integer,Description="LAI">',
        '##INFO=<ID=AF_afr_XX,Number=A,Type=Float,Description="">',
        '##INFO=<ID=AC_afr_XX,Number=A,Type=Integer,Description="">',
        '##INFO=<ID=AN_afr_XX,Number=1,Type=Integer,Description="">',
        '##INFO=<ID=AF_grpmax,Number=A,Type=Float,Description="">',
        "##contig=<ID=chr1,length=1000000>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    recs = [
        "chr1\t100\trs100\tA\tG\t.\tPASS\tAF=0.10;AC=20;AN=200;AF_afr=0.30;AC_afr=30;AN_afr=100;AF_nfe=0.05;AC_nfe=5;AN_nfe=100;AF_afr_tract=0.31;AC_afr_tract=31;AN_afr_tract=100;AF_afr_XX=0.3;AC_afr_XX=15;AN_afr_XX=50;AF_grpmax=0.30",
        "chr1\t110\trs110\tC\tT\t.\tPASS\tAF=0.50;AC=100;AN=200;AF_afr=0.60;AC_afr=60;AN_afr=100;AF_nfe=0.40;AC_nfe=40;AN_nfe=100;AF_afr_tract=0.61;AC_afr_tract=61;AN_afr_tract=100;AF_afr_XX=0.6;AC_afr_XX=30;AN_afr_XX=50;AF_grpmax=0.60",
        "chr1\t120\t.\tG\tA,T\t.\tPASS\tAF=0.20,0.01;AC=40,2;AN=200;AF_afr=0.25,0.02;AC_afr=25,2;AN_afr=100;AF_nfe=0.15,0.0;AC_nfe=15,0;AN_nfe=100;AF_afr_tract=0.25,0.02;AC_afr_tract=25,2;AN_afr_tract=100;AF_afr_XX=0.25,0.0;AC_afr_XX=12,0;AN_afr_XX=50;AF_grpmax=0.25,0.02",
        "chr1\t130\trs130\tT\tC\t.\tAC0\tAF=0.01;AC=2;AN=200;AF_afr=0.02;AC_afr=2;AN_afr=100;AF_nfe=0.0;AC_nfe=0;AN_nfe=100;AF_afr_tract=0.02;AC_afr_tract=2;AN_afr_tract=100;AF_afr_XX=0.0;AC_afr_XX=0;AN_afr_XX=50;AF_grpmax=0.02",
        "chr1\t500000\trs5\tA\tC\t.\tPASS\tAF=0.30;AC=60;AN=200;AF_afr=0.35;AC_afr=35;AN_afr=100;AF_nfe=0.25;AC_nfe=25;AN_nfe=100;AF_afr_tract=0.36;AC_afr_tract=36;AN_afr_tract=100;AF_afr_XX=0.35;AC_afr_XX=17;AN_afr_XX=50;AF_grpmax=0.35",
    ]
    return _write_tabix_vcf(tmp_path, "gnomad_like", header, recs)


@pytest.fixture
def alfa_like_vcf(tmp_path):
    header = [
        "##fileformat=VCFv4.1",
        "##build_id=20260205170148",
        '##FORMAT=<ID=AN,Number=1,Type=Integer,Description="Total allele count">',
        '##FORMAT=<ID=AC,Number=A,Type=Integer,Description="Allele count per ALT">',
        "##contig=<ID=NC_000001.11,length=248956422>",
        "##contig=<ID=NC_000001.10,length=249250621>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMN10492695\tSAMN10492703",
    ]
    recs = [
        "NC_000001.10\t100\trs100\tA\tG\t.\t.\t.\tAN:AC\t10:1\t10:1",          # GRCh37 accession -> dropped
        "NC_000001.11\t100\trs100\tA\tG\t.\t.\t.\tAN:AC\t1000:50\t400:120",
        "NC_000001.11\t110\trs110\tC\tT\t.\t.\t.\tAN:AC\t1000:400\t400:240",
        "NC_000001.11\t120\trs120\tG\tA,T\t.\t.\t.\tAN:AC\t1000:150,10\t400:100,8",
        "NC_000001.11\t130\trs130\tT\tC\t.\t.\t.\tAN:AC\t0:0\t400:4",
    ]
    return _write_tabix_vcf(tmp_path, "alfa_like", header, recs)


@pytest.fixture
def query_variants():
    """variant_df using plain chromosome labels and mixed allele orientations."""
    return pd.DataFrame(
        {"chrom": ["1", "1", "1", "1", "1", "1"],
         "pos": [100, 110, 120, 130, 140, 500000],
         "ref": ["A", "T", "G", "T", "A", "A"],       # rs110 is flipped relative to the VCF (C/T -> T/C)
         "alt": ["G", "C", "T", "C", "T", "C"]},      # pos 140 is absent from the references; A/T ambiguous
        index=["v100", "v110", "v120", "v130", "v140", "v5e5"],
    )


def make_sim_cohort(seed=0, n=30, m=40, n_phen=3):
    """
    Simulated admixed cohort: two-ancestry hard-call panel, dosages drawn from
    ancestry-specific allele frequencies, null phenotypes, and an in-memory
    FrequencyTable holding the true per-ancestry frequencies (group keys 'afr', 'nfe').
    """
    from rfmix_reader import from_legacy
    from localqtl.haplotypeio import LocalAncestryPanel
    from localqtl.freq import FrequencyTable

    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    pos = np.arange(1, m + 1) * 1000
    variant_ids = [f"v{i}" for i in range(m)]
    p_afr = rng.uniform(0.05, 0.95, size=m)
    p_eur = np.clip(p_afr + rng.normal(0, 0.15, size=m), 0.02, 0.98)
    # haplotype ancestries (AFR=0, EUR=1) with 70% AFR
    codes = (rng.uniform(size=(m, n, 2)) > 0.7).astype(int)
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(2)], axis=-1).astype(np.int8)
    alleles = np.where(codes == 0, rng.uniform(size=(m, n, 2)) < p_afr[:, None, None],
                       rng.uniform(size=(m, n, 2)) < p_eur[:, None, None])
    G = alleles.sum(axis=-1).astype(np.float32)
    genotype_df = pd.DataFrame(G, index=variant_ids, columns=samples)
    bases = np.array(list("ACGT"))
    ref = rng.choice(bases, size=m)
    alt = np.array([rng.choice([b for b in "ACGT" if b != r]) for r in ref])
    variant_df = pd.DataFrame({"chrom": "chr1", "pos": pos, "ref": ref, "alt": alt}, index=variant_ids)
    Y = rng.normal(size=(n_phen, n)).astype(np.float32)
    phenotype_df = pd.DataFrame(Y, index=[f"gene{i}" for i in range(n_phen)], columns=samples)
    phenotype_pos_df = pd.DataFrame({"chr": "chr1", "start": [5000, 20000, 35000][:n_phen],
                                     "end": [5000, 20000, 35000][:n_phen]}, index=phenotype_df.index)
    loci = pd.DataFrame({"chromosome": "chr1", "physical_position": pos})
    g_anc = pd.DataFrame({"sample_id": samples, "P0": 0.7, "P1": 0.3, "chrom": "1"})
    panel = LocalAncestryPanel(from_legacy(loci, g_anc, admix))
    table = FrequencyTable(pd.DataFrame({
        "chrom": "chr1", "pos": pos, "ref": ref, "alt": alt,
        "af_afr": p_afr, "ac_afr": np.round(p_afr * 2000), "an_afr": 2000,
        "af_nfe": p_eur, "ac_nfe": np.round(p_eur * 5000), "an_nfe": 5000,
    }), source="sim")
    return dict(genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, panel=panel, admix=admix, table=table,
                p_afr=p_afr, p_eur=p_eur, samples=samples, rng=rng)


@pytest.fixture
def sim_cohort():
    return make_sim_cohort
