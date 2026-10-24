import pandas as pd

from localqtl.compat import from_tensorqtl_columns, to_tensorqtl_columns


def test_round_trip_column_names():
    df = pd.DataFrame(columns=["phenotype_id", "true_dof", "pval_true_dof", "se_g", "se_gi", "se_g-age", "slope_se"])
    tq = to_tensorqtl_columns(df)
    assert list(tq.columns) == ["phenotype_id", "true_df", "pval_true_df", "b_g_se", "b_gi_se", "b_g-age_se", "slope_se"]
    back = from_tensorqtl_columns(tq)
    assert list(back.columns) == list(df.columns)
    # tensorQTL's older minor-allele spelling is normalised
    assert list(from_tensorqtl_columns(pd.DataFrame(columns=["minor_allele_samples"])).columns) == ["ma_samples"]
