"""
tensorQTL naming compatibility.

localQTL's result columns follow tensorQTL except for a few names that were
spelled out more explicitly (``true_dof`` instead of ``true_df``).  The
helpers here translate in both directions so that files written by one tool
can be consumed by the other (``annotate_genes``, ``get_significant_pairs``,
pyqtl, ...).
"""
from __future__ import annotations

import pandas as pd

__all__ = [
    "TENSORQTL_COLUMN_ALIASES",
    "LOCALQTL_COLUMN_ALIASES",
    "to_tensorqtl_columns",
    "from_tensorqtl_columns",
]

# localQTL name -> tensorQTL name
TENSORQTL_COLUMN_ALIASES: dict[str, str] = {
    "true_dof": "true_df",
    "pval_true_dof": "pval_true_df",
    "minor_allele_samples": "ma_samples",
    "minor_allele_count": "ma_count",
}
# tensorQTL name -> localQTL name
LOCALQTL_COLUMN_ALIASES: dict[str, str] = {v: k for k, v in TENSORQTL_COLUMN_ALIASES.items()
                                           if not k.startswith("minor_allele")}
LOCALQTL_COLUMN_ALIASES.update({"minor_allele_samples": "ma_samples",
                                "minor_allele_count": "ma_count"})


def to_tensorqtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename localQTL result columns to their tensorQTL spelling (copy): the
    aliases above plus the interaction standard errors ``se_<term>`` ->
    ``b_<term>_se`` (``se_g``, ``se_i``, ``se_gi``, ``se_<name>``, ``se_g-<name>``).
    """
    mapping = {k: v for k, v in TENSORQTL_COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    for c in df.columns:
        if isinstance(c, str) and c.startswith("se_") and c not in mapping and c != "se_":
            tgt = f"b_{c[3:]}_se"
            if tgt not in df.columns:
                mapping[c] = tgt
    return df.rename(columns=mapping)


def from_tensorqtl_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename tensorQTL result columns to the localQTL spelling (copy), inverse of :func:`to_tensorqtl_columns`."""
    mapping = {k: v for k, v in LOCALQTL_COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    for c in df.columns:
        if isinstance(c, str) and c.startswith("b_") and c.endswith("_se") and len(c) > 5 and c not in mapping:
            tgt = f"se_{c[2:-3]}"
            if tgt not in df.columns:
                mapping[c] = tgt
    return df.rename(columns=mapping)
