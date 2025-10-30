import numpy as np
import pandas as pd
import pytest

import src.localqtl.phenotypeio as ph

def test_gpu_available_returns_bool(monkeypatch):
    monkeypatch.setattr("phenotypeio.cp", None)  # pretend no cupy
    assert not ph.gpu_available()

def test_read_phenotype_bed_dataframe(tmp_path):
    bedfile = tmp_path / "toy.bed"
    df = pd.DataFrame({
        "#chr": ["1","1"],
        "start": [100,200],
        "end": [100,200],
        "id": ["g1","g2"],
        "s1": [1.0,2.0],
        "s2": [2.0,3.0]
    })
    df.to_csv(bedfile, sep="\t", index=False)
    
    pheno, pos = ph.read_phenotype_bed(str(bedfile))
    assert "g1" in pheno.index
    assert "chr" in pos.columns

def test_read_phenotype_bed_tensor_cpu(tmp_path):
    bedfile = tmp_path / "toy.bed"
    df = pd.DataFrame({
        "#chr": ["1"],
        "start": [100],
        "end": [100],
        "id": ["g1"],
        "s1": [1.0],
        "s2": [2.0]
    })
    df.to_csv(bedfile, sep="\t", index=False)
    
    arr, pos = ph.read_phenotype_bed(str(bedfile), as_tensor=True, device="cpu")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,2)
