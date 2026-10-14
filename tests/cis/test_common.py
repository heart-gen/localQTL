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


def test_residualize_batch_handles_multiple_channels_per_sample():
    """Each (variant, channel) column of H must be residualised across samples
    independently; a flat reshape of (m, n, p) would mix samples and channels."""
    import torch
    from src.localqtl.regression_kernels import Residualizer
    from src.localqtl.cis.common import (
        residualize_batch, residualize_batch_interaction, residualize_batch_dosage,
    )

    torch.manual_seed(0)
    m, n, p = 3, 12, 2
    C = torch.randn(n, 2)
    rez = Residualizer(C)
    y = torch.randn(n)
    G = torch.randn(m, n)
    H = torch.rand(m, n, p) * 2
    _, _, H_resid = residualize_batch(y, G, H, rez, center=True)
    for v in range(m):
        for k in range(p):
            (ref,) = rez.transform(H[v, :, k].unsqueeze(0), center=True)
            torch.testing.assert_close(H_resid[v, :, k], ref.squeeze(0), rtol=1e-5, atol=1e-5)

    _, _, H_res2, I_resid = residualize_batch_interaction(y, G, H, rez, center=True)
    torch.testing.assert_close(H_res2, H_resid, rtol=1e-5, atol=1e-5)
    (ref_i,) = rez.transform((G[1] * H[1, :, 1]).unsqueeze(0), center=True)
    torch.testing.assert_close(I_resid[1, :, 1], ref_i.squeeze(0), rtol=1e-5, atol=1e-5)

    L = H / H.sum(dim=2, keepdim=True) * 2          # expected counts summing to 2
    _, D_resid, L_resid = residualize_batch_dosage(y, G, L, rez, center=True)
    assert D_resid.shape == (m, n, p) and L_resid.shape == (m, n, p - 1)
    (ref_d,) = rez.transform((G[2] * L[2, :, 0] / 2).unsqueeze(0), center=True)
    torch.testing.assert_close(D_resid[2, :, 0], ref_d.squeeze(0), rtol=1e-5, atol=1e-5)
