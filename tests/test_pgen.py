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
