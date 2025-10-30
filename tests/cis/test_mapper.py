import pandas as pd
import pytest

from src.localqtl.cis.mapper import CisMapper
from src.localqtl.utils import SimpleLogger


def _build_mapper(toy_data):
    return CisMapper(
        genotype_df=toy_data["genotype_df"],
        variant_df=toy_data["variant_df"],
        phenotype_df=toy_data["phenotype_df"],
        phenotype_pos_df=toy_data["phenotype_pos_df"],
        covariates_df=pd.DataFrame({"cov": [0.0, 1.0, 2.0, 3.0]}, index=toy_data["samples"]),
        device="cpu",
        logger=SimpleLogger(verbose=False),
    )


def test_cis_mapper_delegates_nominal(monkeypatch, toy_data):
    mapper = _build_mapper(toy_data)
    called = {}

    def fake_nominal(**kwargs):
        called.update(kwargs)
        return pd.DataFrame({"phenotype_id": ["geneA"]})

    monkeypatch.setattr("src.localqtl.cis.mapper._map_nominal", fake_nominal)

    result = mapper.map_nominal(nperm=2, maf_threshold=0.5)

    assert not result.empty
    assert called["nperm"] == 2
    assert called["maf_threshold"] == 0.5
    assert called["device"] == "cpu"


def test_cis_mapper_delegates_permutations(monkeypatch, toy_data):
    mapper = _build_mapper(toy_data)
    called = {}

    def fake_perm(**kwargs):
        called.update(kwargs)
        return pd.DataFrame({"phenotype_id": ["geneA"], "pval_beta": [0.01]})

    monkeypatch.setattr("src.localqtl.cis.mapper._map_permutations", fake_perm)

    result = mapper.map_permutations(nperm=7, maf_threshold=0.3, seed=123)

    assert not result.empty
    assert called["nperm"] == 7
    assert called["maf_threshold"] == 0.3
    assert called["seed"] == 123


def test_cis_mapper_delegates_independent(monkeypatch, toy_data):
    mapper = _build_mapper(toy_data)
    called = {}

    def fake_independent(**kwargs):
        called.update(kwargs)
        return pd.DataFrame({"phenotype_id": ["geneA"], "rank": [1]})

    monkeypatch.setattr("src.localqtl.cis.mapper._map_independent", fake_independent)

    cis_df = pd.DataFrame({
        "phenotype_id": ["geneA"],
        "variant_id": ["v1_chr1_100"],
        "pval_beta": [0.01],
        "qval": [0.01],
    })

    result = mapper.map_independent(cis_df=cis_df, fdr=0.1, maf_threshold=0.2)

    assert not result.empty
    assert called["fdr"] == 0.1
    assert called["maf_threshold"] == 0.2
    assert called["cis_df"].equals(cis_df)


def test_cis_mapper_calculates_qvalues(monkeypatch, toy_data):
    mapper = _build_mapper(toy_data)
    called = {}

    def fake_calc(res_df, fdr, qvalue_lambda, logger):
        called["res_df"] = res_df
        called["fdr"] = fdr
        called["qvalue_lambda"] = qvalue_lambda
        called["logger"] = logger
        return res_df.assign(qval=[0.02])

    monkeypatch.setattr("src.localqtl.cis.mapper._calculate_qvalues", fake_calc)

    perm_df = pd.DataFrame({
        "phenotype_id": ["geneA"],
        "pval_beta": [0.01],
    })

    result = mapper.calculate_qvalues(perm_df, fdr=0.1, qvalue_lambda=0.4)

    assert not result.empty
    assert called["res_df"].equals(perm_df)
    assert called["fdr"] == 0.1
    assert called["qvalue_lambda"] == 0.4
    assert called["logger"] is mapper.logger
