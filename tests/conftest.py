import numpy as np
import pandas as pd
import pytest

############################
# Core synthetic test data #
############################

def make_test_data():
    # ---------- Samples ----------
    samples = ["S1", "S2", "S3", "S4"]

    # ---------- Variant metadata ----------
    variant_index = [
        "v1_chr1_100",
        "v2_chr1_200",
        "v3_chr1_300",
        "v4_chr1_800",
        "v5_chr1_900",
        "v1_chr2_150",
        "v2_chr2_250",
        "v3_chr2_400",
    ]

    chroms = ["1","1","1","1","1","2","2","2"]
    poses  = [100, 200, 300, 800, 900, 150, 250, 400]

    variant_df = pd.DataFrame({
        "chrom": chroms,
        "pos": poses,
    }, index=variant_index)

    # ---------- Genotypes ----------
    # shape: variants x samples (8 x 4)
    # We'll deliberately include a missing code -9 in some rows
    genotype_mat = np.array([
        [0, 1, 2, 0],   # v1_chr1_100
        [1, 1, 2, 2],   # v2_chr1_200
        [2, 2, 2,-9],   # v3_chr1_300  <-- has missing
        [0, 0, 1, 1],   # v4_chr1_800
        [2, 1, 1, 0],   # v5_chr1_900
        [0, 0, 0, 0],   # v1_chr2_150
        [1, 2, 1, 1],   # v2_chr2_250
        [2, 2, 1, 1],   # v3_chr2_400
    ], dtype=np.float32)

    genotype_df = pd.DataFrame(
        genotype_mat,
        index=variant_index,
        columns=samples
    )

    # ---------- Phenotypes + positions ----------
    # We'll include one constant phenotype that should get dropped.
    phenotype_index = ["geneA", "geneB", "geneC", "geneConst"]

    phenotype_mat = np.array([
        [ 10.0,  11.0,  12.0,  13.0],  # geneA  (varies)
        [  5.0,   5.5,   6.0,   6.5],  # geneB  (varies)
        [100.0, 110.0, 105.0, 120.0],  # geneC  (varies)
        [ 42.0,  42.0,  42.0,  42.0],  # geneConst (constant -> should drop)
    ], dtype=np.float32)

    phenotype_df = pd.DataFrame(
        phenotype_mat,
        index=phenotype_index,
        columns=samples
    )

    # phenotype_pos_df in "start/end" format
    phenotype_pos_df = pd.DataFrame({
        "chr":  ["1",  "1",  "2",  "1"],
        "start":[180, 850, 260, 500],
        "end":  [180, 900, 260, 500],
    }, index=phenotype_index)

    return genotype_df, variant_df, phenotype_df, phenotype_pos_df

###########################
# Pytest fixtures version #
###########################

@pytest.fixture
def toy_data():
    """
    Returns a dict with all basic toy dataframes needed to test genotypeio.
    """
    (genotype_df,
     variant_df,
     phenotype_df,
     phenotype_pos_df) = make_test_data()

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "samples": list(genotype_df.columns),
    }

def _make_loci_df(chroms, positions, n_pops):
    """
    chroms: list[str] length = n_loci
    positions: list[int] length = n_loci
    n_pops: number of ancestries

    Returns:
      loci_df (pd.DataFrame):
        index: hap IDs like "1_100_A0"
        columns: chrom, pos, ancestry, index
        'index' is 0..n_loci-1 for pop0, then 0..n_loci-1 for pop1, etc.
        (this is similar in spirit to haplotypeio.RFMixReader for n_pops>2,
         and a simplified version for n_pops==2)
    """
    records = []
    n_loci = len(chroms)
    for anc in range(n_pops):
        for i in range(n_loci):
            hap_id = f"{chroms[i]}_{positions[i]}_A{anc}"
            records.append({
                "hap": hap_id,
                "chrom": chroms[i],
                "pos": positions[i],
                "ancestry": anc,
                # This matches the pattern where index can be offset by ancestry
                "index": anc * n_loci + i,
            })
    loci_df = pd.DataFrame.from_records(records).set_index("hap")
    return loci_df


@pytest.fixture
def toy_hap_data_2anc(toy_data):
    """
    Build small, deterministic haplotype/local ancestry data for 2 ancestries.
    Mirrors toy_data's variant_df rows (7 loci across chr1/chr2).

    We create:
      haplotypes: shape (7 loci, 4 samples, 2 ancestries)
      loci_df: metadata with hap IDs per (locus, ancestry)
      plus we forward genotype_df/etc from toy_data so tests
      can directly build InputGeneratorCisWithHaps.
    """
    genotype_df      = toy_data["genotype_df"]
    variant_df       = toy_data["variant_df"]
    phenotype_df     = toy_data["phenotype_df"]
    phenotype_pos_df = toy_data["phenotype_pos_df"]

    # loci: same order as variant_df rows
    chroms = variant_df["chrom"].tolist()   # ["1","1","1","1","1","2","2"]
    positions = variant_df["pos"].tolist()  # [100,200,300,800,900,150,250]

    n_loci    = len(positions)  # 7
    n_samples = genotype_df.shape[1]  # should be 4
    n_pops    = 2  # two ancestries

    # Shape: (loci, samples, ancestries)
    # We'll make a simple deterministic pattern:
    # ancestry0 = locus_index + sample_index
    # ancestry1 = locus_index + 10*sample_index
    hap_anc0 = np.zeros((n_loci, n_samples), dtype=float)
    hap_anc1 = np.zeros((n_loci, n_samples), dtype=float)
    for i in range(n_loci):
        for s in range(n_samples):
            hap_anc0[i, s] = i + s       # e.g. 0+0,0+1,0+2,0+3; 1+0,1+1,...
            hap_anc1[i, s] = i + 10*s    # e.g. 0+0,0+10,0+20,0+30; 1+0,...

    hap = np.stack([hap_anc0, hap_anc1], axis=2)  # (loci, samples, 2)

    # inject some NaNs for interpolation test:
    # say locus 2, sample 1, ancestry0 is missing
    hap[2, 1, 0] = np.nan
    # also locus 5, sample 3, ancestry1 missing
    hap[5, 3, 1] = np.nan

    loci_df = _make_loci_df(chroms, positions, n_pops=2)

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "haplotypes": hap,     # numpy ndarray (loci x samples x 2)
        "loci_df": loci_df,    # pandas DataFrame
    }


@pytest.fixture
def toy_hap_data_3anc(toy_data):
    """
    Same idea as toy_hap_data_2anc, but now with 3 ancestries.
    Used to test n_pops > 2 behavior and interpolation.

    We don't strictly need phenotype_df here, but we include it for symmetry.
    """
    genotype_df      = toy_data["genotype_df"]
    variant_df       = toy_data["variant_df"]
    phenotype_df     = toy_data["phenotype_df"]
    phenotype_pos_df = toy_data["phenotype_pos_df"]

    chroms = variant_df["chrom"].tolist()
    positions = variant_df["pos"].tolist()

    n_loci    = len(positions)
    n_samples = genotype_df.shape[1]  # 4
    n_pops    = 3

    # ancestry0 = i + s
    # ancestry1 = i + 10*s
    # ancestry2 = i + 100*s
    hap_ancs = []
    for anc in range(n_pops):
        anc_mat = np.zeros((n_loci, n_samples), dtype=float)
        for i in range(n_loci):
            for s in range(n_samples):
                anc_mat[i, s] = i + (10**anc)*s
        hap_ancs.append(anc_mat)
    hap = np.stack(hap_ancs, axis=2)  # (loci, samples, 3)

    # Add a couple NaNs to test interpolation
    hap[1, 2, 0] = np.nan  # ancestry0 missing at (locus1, sample2)
    hap[4, 0, 2] = np.nan  # ancestry2 missing at (locus4, sample0)

    loci_df = _make_loci_df(chroms, positions, n_pops=3)

    return {
        "genotype_df": genotype_df,
        "variant_df": variant_df,
        "phenotype_df": phenotype_df,
        "phenotype_pos_df": phenotype_pos_df,
        "haplotypes": hap,     # numpy ndarray (loci x samples x 3)
        "loci_df": loci_df,    # pandas DataFrame with ancestry column 0..2
    }



@pytest.fixture
def toy_panel_2anc(toy_data):
    """
    The toy variant grid as a localqtl.haplotypeio.LocalAncestryPanel (two ancestries,
    valid diploid counts), plus the (loci, samples, 2) count array it was built from.
    """
    from rfmix_reader import from_legacy
    from localqtl.haplotypeio import LocalAncestryPanel

    var_df = toy_data["variant_df"]
    samples = list(toy_data["genotype_df"].columns)
    loci = pd.DataFrame({"chromosome": var_df["chrom"].astype(str).values,
                         "physical_position": var_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": samples, "P0": 0.5, "P1": 0.5, "chrom": "1"})
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 2, size=(len(loci), len(samples), 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(2)], axis=-1).astype(np.int8)
    return {**toy_data, "panel": LocalAncestryPanel(from_legacy(loci, g_anc, admix)), "admix": admix}


@pytest.fixture
def toy_panel_2anc_posterior(toy_panel_2anc):
    """
    ``toy_panel_2anc`` with soft per-haplotype posteriors: the called ancestry
    gets mass 0.6-0.95, the other the remainder, so the hard calls (argmax)
    are unchanged but the expected counts are fractional.
    """
    from rfmix_reader import build_dataset
    from localqtl.haplotypeio import LocalAncestryPanel

    ds = toy_panel_2anc["panel"].ds
    codes = np.asarray(ds.la.haplotypes.values)                     # (v, s, 2)
    rng = np.random.default_rng(11)
    conf = rng.uniform(0.6, 0.95, size=codes.shape).astype(np.float32)
    post = np.zeros(codes.shape + (2,), dtype=np.float32)          # (v, s, 2, A)
    for a in range(2):
        post[..., a] = np.where(codes == a, conf, 1.0 - conf)
    ds_post = build_dataset(
        ds.chromosome.values, ds.variant_position.values, ds.segment_end.values, codes,
        sample_id=ds.la.samples, ancestry=ds.la.ancestries, posterior=post,
    )
    return {**toy_panel_2anc, "panel_post": LocalAncestryPanel(ds_post, use_posterior=True),
            "panel_hard": LocalAncestryPanel(ds_post), "posterior": post}


def make_random_cohort(seed=0, n=40, m=6, chrom="chr1", effect=0.4):
    """Random dosages / one phenotype with a planted effect on the first variant."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    G = rng.integers(0, 3, size=(m, n)).astype(np.float32)
    variant_ids = [f"v{i}_{chrom}_{100 + 10 * i}" for i in range(m)]
    genotype_df = pd.DataFrame(G, index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({"chrom": chrom, "pos": [100 + 10 * i for i in range(m)]}, index=variant_ids)
    y = effect * G[0] + rng.normal(size=n)
    phenotype_df = pd.DataFrame([y], index=["geneA"], columns=samples).astype(np.float32)
    phenotype_pos_df = pd.DataFrame({"chr": [chrom], "start": [120], "end": [120]}, index=["geneA"])
    return dict(genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, samples=samples, G=G, y=y, rng=rng)


@pytest.fixture
def random_cohort():
    """Factory fixture: ``random_cohort(seed=..., n=..., m=...)``."""
    return make_random_cohort


def ols_fit(y, X):
    """numpy OLS with intercept -> (betas, ses, dof) for the columns of X (intercept excluded)."""
    X1 = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=np.float64)])
    b = np.linalg.lstsq(X1, np.asarray(y, dtype=np.float64), rcond=None)[0]
    r = y - X1 @ b
    dof = len(y) - X1.shape[1]
    cov = (r @ r) / dof * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    return b[1:], se[1:], dof


def make_random_panel(variant_df, samples, seed=3, n_anc=2):
    """In-memory LocalAncestryPanel (hard calls) on the rows of ``variant_df``; returns (panel, admix)."""
    from rfmix_reader import from_legacy
    from localqtl.haplotypeio import LocalAncestryPanel
    rng = np.random.default_rng(seed)
    loci = pd.DataFrame({"chromosome": variant_df["chrom"].astype(str).values,
                         "physical_position": variant_df["pos"].values})
    g_anc = pd.DataFrame({"sample_id": list(samples), **{f"P{a}": 1.0 / n_anc for a in range(n_anc)}, "chrom": "1"})
    codes = rng.integers(0, n_anc, size=(len(variant_df), len(samples), 2))
    admix = np.stack([(codes == a).sum(axis=-1) for a in range(n_anc)], axis=-1).astype(np.int8)
    return LocalAncestryPanel(from_legacy(loci, g_anc, admix)), admix


def write_toy_pgen(prefix, genotype_df, variant_df):
    """Write ``genotype_df`` (hard calls, -9 missing) as PLINK 2 pgen/psam/pvar at ``prefix``."""
    import pgenlib as pg
    m, n = genotype_df.shape
    w = pg.PgenWriter(f"{prefix}.pgen".encode(), n, m, False)
    for row in genotype_df.to_numpy():
        r = np.asarray(row, dtype=np.int8)
        r[r < 0] = -9
        w.append_biallelic(r)
    w.close()
    with open(f"{prefix}.psam", "w") as f:
        f.write("#IID\tSEX\n")
        for s in genotype_df.columns:
            f.write(f"{s}\tNA\n")
    with open(f"{prefix}.pvar", "w") as f:
        f.write("#CHROM\tPOS\tID\tREF\tALT\n")
        ref = variant_df["ref"] if "ref" in variant_df else pd.Series("A", index=variant_df.index)
        alt = variant_df["alt"] if "alt" in variant_df else pd.Series("G", index=variant_df.index)
        for vid, c, p in zip(variant_df.index, variant_df["chrom"], variant_df["pos"]):
            f.write(f"{c}\t{int(p)}\t{vid}\t{ref[vid]}\t{alt[vid]}\n")
    return prefix


def write_phenotype_bed(path, phenotype_df, phenotype_pos_df):
    """Write a tensorQTL-style phenotype BED (``#chr start end id samples...``; start is 0-based)."""
    pos = phenotype_pos_df.copy()
    if "pos" in pos.columns:
        pos["start"] = pos["pos"] - 1
        pos["end"] = pos["pos"]
    else:
        pos["start"] = pos["start"] - 1
    bed = pd.DataFrame({"#chr": pos["chr"].astype(str).values, "start": pos["start"].values,
                        "end": pos["end"].values, "id": phenotype_df.index.values})
    bed = pd.concat([bed.reset_index(drop=True), phenotype_df.reset_index(drop=True)], axis=1)
    bed = bed.sort_values(["#chr", "start", "end"])
    bed.to_csv(path, sep="\t", index=False)
    return path


def make_cohort_with_chroms(seed=5, n=60, m_per_chrom=(8, 6), n_pheno_per_chrom=(2, 1), effect=0.6):
    """Random cohort over several chromosomes, phenotypes sorted by position, planted effects."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(n)]
    rows, chroms, poss = [], [], []
    for ci, m in enumerate(m_per_chrom, 1):
        for i in range(m):
            rows.append(f"v{ci}_{i}")
            chroms.append(f"chr{ci}")
            poss.append(1000 + 500 * i)
    G = rng.integers(0, 3, size=(len(rows), n)).astype(np.float32)
    genotype_df = pd.DataFrame(G, index=rows, columns=samples)
    variant_df = pd.DataFrame({"chrom": chroms, "pos": poss, "ref": "A", "alt": "G"}, index=rows)
    pids, pchr, ppos, Y = [], [], [], []
    for ci, npheno in enumerate(n_pheno_per_chrom, 1):
        vrows = [k for k, c in enumerate(chroms) if c == f"chr{ci}"]
        for j in range(npheno):
            pids.append(f"gene{ci}_{j}")
            pchr.append(f"chr{ci}")
            ppos.append(1200 + 700 * j)
            Y.append(effect * G[vrows[j]] + rng.normal(size=n))
    phenotype_df = pd.DataFrame(np.array(Y, dtype=np.float32), index=pids, columns=samples)
    phenotype_pos_df = pd.DataFrame({"chr": pchr, "pos": ppos}, index=pids)
    covariates_df = pd.DataFrame(rng.normal(size=(n, 2)), index=samples, columns=["PC1", "PC2"])
    return dict(genotype_df=genotype_df, variant_df=variant_df, phenotype_df=phenotype_df,
                phenotype_pos_df=phenotype_pos_df, covariates_df=covariates_df, samples=samples, rng=rng)


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="run tests that call external APIs (IGVF Catalog, AlphaGenome Atlas)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live (external API)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
