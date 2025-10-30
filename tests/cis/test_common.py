import numpy as np
import pandas as pd
import torch

from src.localqtl.cis.common import (
    dosage_vector_for_covariate,
    residualize_matrix_with_covariates,
    residualize_batch,
)
from src.localqtl.regression_kernels import Residualizer


def test_dosage_vector_for_covariate_handles_reordering_and_imputation(toy_data):
    genotype_df = toy_data["genotype_df"]
    samples = toy_data["samples"]

    # introduce a non-default sample order and confirm we respect it
    shuffled_samples = list(reversed(samples))
    row = dosage_vector_for_covariate(
        genotype_df=genotype_df,
        variant_id="v3_chr1_300",
        sample_order=pd.Index(shuffled_samples),
        missing=-9.0,
    )

    assert row.shape == (len(samples),)
    # ensure mean-imputation replaced the -9.0 entry with the average of the observed values
    observed = genotype_df.loc["v3_chr1_300", shuffled_samples].to_numpy()
    mask = observed == -9.0
    expected_mean = observed[~mask].mean(dtype=np.float32)
    assert np.isclose(row[mask][0], expected_mean)
    # and all remaining entries match the requested order
    assert np.allclose(row[~mask], observed[~mask])


def test_residualize_matrix_with_covariates_centers_rows():
    Y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    cov = pd.DataFrame({"c1": [1.0, 2.0, 3.0]}, index=["s1", "s2", "s3"])

    Y_resid, rez = residualize_matrix_with_covariates(Y, cov, device="cpu")

    assert rez is not None
    assert torch.allclose(Y_resid.sum(dim=1), torch.zeros(2), atol=1e-6)

    # Without covariates the tensor should be returned unchanged
    Y_passthrough, rez_none = residualize_matrix_with_covariates(Y, None, device="cpu")
    assert rez_none is None
    assert torch.allclose(Y_passthrough, Y)


def test_residualize_batch_supports_grouped_and_scalar_inputs():
    rez = Residualizer(torch.ones((3, 1)))
    G = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    y = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    y_resid, G_resid, H_resid = residualize_batch(y, G, None, rez, center=True, group=False)
    assert H_resid is None
    assert y_resid.shape == (3,)
    assert G_resid.shape == G.shape
    assert torch.allclose(y_resid.sum(), torch.tensor(0.0), atol=1e-6)

    grouped = [torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32),
               torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)]
    y_list, G_group_resid, _ = residualize_batch(grouped, G, None, rez, center=True, group=True)
    assert len(y_list) == 2
    for vec in y_list:
        assert vec.shape == (3,)
        assert torch.allclose(vec.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(G_group_resid, G_resid)
