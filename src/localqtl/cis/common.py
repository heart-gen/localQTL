import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .._torch_utils import move_to_device, resolve_device
from ..regression_kernels import Residualizer

__all__ = [
    "dosage_vector_for_covariate",
    "residualize_matrix_with_covariates",
    "residualize_batch",
    "residualize_batch_interaction",
    "prepare_interaction_covariate",
    "align_like_casefold",
]


def _coerce_tensor(value, device: torch.device) -> torch.Tensor:
    return move_to_device(value, device) if torch.is_tensor(value) else torch.as_tensor(
        value, dtype=torch.float32, device=device
    )


def _make_gen(seed: int | None, device: str) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & ((1 << 63) - 1))
    return g

def dosage_vector_for_covariate(
    genotype_df: pd.DataFrame,
    variant_id: str,
    sample_order: pd.Index,
    missing: float | int | None,
) -> np.ndarray:
    """Fetch a dosage row aligned to samples; impute 'missing' to mean of observed."""
    G_aligned = align_like_casefold(
        genotype_df,
        like=sample_order,
        axis="columns",
        what="sample IDs in genotype_df.columns",
        strict=True,
    )

    row = G_aligned.loc[variant_id].to_numpy(dtype=np.float32, copy=True)
    if missing is not None:
        mm = (row == missing)
        if mm.any():
            if (~mm).any():
                row[mm] = row[~mm].mean(dtype=np.float32)
            else:
                row[mm] = 0.0
    return row


def residualize_matrix_with_covariates(
        Y: torch.Tensor, C: Optional[pd.DataFrame], device: str,
        tensorqtl_flavor: bool = False
) -> Tuple[torch.Tensor, Optional[Residualizer]]:
    """
    Residualize (features x samples) matrix Y against covariates C across samples.
    Returns residualized Y and a Residualizer (or None if no covariates).
    """
    if C is None:
        return Y, None
    dev = resolve_device(device)
    C_t = torch.tensor(C.values, dtype=torch.float32, device=dev)
    rez = Residualizer(C_t, tensorqtl_flavor=tensorqtl_flavor)
    (Y_resid,) = rez.transform(move_to_device(Y, dev), center=True)
    return Y_resid, rez


def residualize_batch(
        y, G: torch.Tensor, H: Optional[torch.Tensor],
        rez: Optional[Residualizer], center: bool = True, group: bool = False,
) -> Tuple[object, torch.Tensor, Optional[torch.Tensor]]:
    """
    Residualize y, G, and optional H with the same Residualizer.
    - If group=False: y is a single vector (n,), returns (y_resid 1D, G_resid, H_resid).
    - If group=True:  y is a stack (k x n) or list of length k, returns (list_of_k 1D tensors, G_resid, H_resid).
    """
    if rez is not None and hasattr(rez, "Q_t") and getattr(rez.Q_t, "numel", lambda: 0)() > 0:
        dev = resolve_device(rez.Q_t.device)
    else:
        dev = resolve_device(G.device)
    G = move_to_device(G, dev)
    if H is not None:
        H = move_to_device(H, dev)

    if rez is None:
        # Pass-through; normalize return type for group mode
        if group:
            if isinstance(y, (list, tuple)):
                y_list = [torch.as_tensor(yi) if not torch.is_tensor(yi) else yi
                          for yi in y]
            else:
                Y = y if torch.is_tensor(y) else torch.as_tensor(y)
                y_list = [Y] if Y.ndim == 1 else [Y[i, :] for i in range(Y.shape[0])]
            return y_list, G, H
        return (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)), G, H

    # Build Y on the same device as G/rez
    if group:
        if isinstance(y, (list, tuple)):
            Y = torch.stack([_coerce_tensor(yi, dev) for yi in y], dim=0)
        else:
            Y = _coerce_tensor(y, dev)
            if Y.ndim == 1:
                Y = Y.unsqueeze(0)
    else:
        Y = _coerce_tensor(y, dev)
        if Y.ndim == 1:
            Y = Y.unsqueeze(0)
    
    # Prepare matrices for one-pass residualization
    mats: List[torch.Tensor] = [G]
    H_shape = None
    if H is not None:
        m, n, pH = H.shape
        H_shape = (m, n, pH)
        mats.append(H.reshape(m * pH, n))

    mats_resid = rez.transform(*mats, Y, center=center)

    G_resid = mats_resid[0]
    idx, H_resid = 1, None
    if H is not None:
        H_resid = mats_resid[idx].reshape(H_shape)
        idx += 1
    Y_resid = mats_resid[idx]

    if group:
        return [Y_resid[i, :] for i in range(Y_resid.shape[0])], G_resid, H_resid
    else:
        return Y_resid.squeeze(0), G_resid, H_resid


def residualize_batch_interaction(
        y, G: torch.Tensor, H: torch.Tensor,
        rez: Optional[Residualizer], center: bool = True, group: bool = False,
) -> Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Residualize y, G, H, and interaction I=G*H with the same Residualizer.
    Returns (y_resid, G_resid, H_resid, I_resid).
    """
    if H is None:
        raise ValueError("H is required for interaction residualization.")

    if rez is not None and hasattr(rez, "Q_t") and getattr(rez.Q_t, "numel", lambda: 0)() > 0:
        dev = resolve_device(rez.Q_t.device)
    else:
        dev = resolve_device(G.device)
    G = move_to_device(G, dev)
    H = move_to_device(H, dev)

    if rez is None:
        if group:
            if isinstance(y, (list, tuple)):
                y_list = [torch.as_tensor(yi) if not torch.is_tensor(yi) else yi
                          for yi in y]
            else:
                Y = y if torch.is_tensor(y) else torch.as_tensor(y)
                y_list = [Y] if Y.ndim == 1 else [Y[i, :] for i in range(Y.shape[0])]
            I = G.unsqueeze(-1) * H
            return y_list, G, H, I
        Y = y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)
        I = G.unsqueeze(-1) * H
        return Y, G, H, I

    if group:
        if isinstance(y, (list, tuple)):
            Y = torch.stack([_coerce_tensor(yi, dev) for yi in y], dim=0)
        else:
            Y = _coerce_tensor(y, dev)
            if Y.ndim == 1:
                Y = Y.unsqueeze(0)
    else:
        Y = _coerce_tensor(y, dev)
        if Y.ndim == 1:
            Y = Y.unsqueeze(0)

    m, n, pH = H.shape
    I = G.unsqueeze(-1) * H  # (m, n, pH)

    mats: List[torch.Tensor] = [G]
    H_shape = (m, n, pH)
    I_shape = (m, n, pH)
    mats.append(H.reshape(m * pH, n))
    mats.append(I.reshape(m * pH, n))

    mats_resid = rez.transform(*mats, Y, center=center)
    G_resid = mats_resid[0]
    idx = 1
    H_resid = mats_resid[idx].reshape(H_shape)
    idx += 1
    I_resid = mats_resid[idx].reshape(I_shape)
    idx += 1
    Y_resid = mats_resid[idx]

    if group:
        return [Y_resid[i, :] for i in range(Y_resid.shape[0])], G_resid, H_resid, I_resid
    return Y_resid.squeeze(0), G_resid, H_resid, I_resid


def align_like_casefold(
    df: pd.DataFrame, like: pd.Index, axis: str = "index",
    what: str = "samples", strict: bool = True,
) -> pd.DataFrame:
    """
    Case-insensitive, order-preserving alignment of df to 'like'.

    axis='index'  -> align rows (df.index) to 'like'
    axis='columns'-> align columns (df.columns) to 'like'
    """
    if axis not in ("index", "columns"):
        raise ValueError("axis must be 'index' or 'columns'")

    src = pd.Index(getattr(df, axis).astype(str))
    src_cf = src.str.casefold()

    dup_mask = src_cf.duplicated(keep=False)
    if dup_mask.any():
        dups = list(src[dup_mask].unique())
        raise ValueError(
            f"Case-insensitive duplicate {what} in df.{axis}: {dups}. "
            "Disambiguate these IDs (they collide when case is ignored)."
        )

    lut = pd.Series(src.values, index=src_cf, dtype=object)
    want_cf = pd.Index(like.astype(str)).str.casefold()
    take = lut.reindex(want_cf)

    if strict:
        missing = want_cf[take.isna()]
        if len(missing):
            raise KeyError(
                f"{len(missing)} {what} not found (case-insensitive): {list(pd.Index(missing).unique())}"
            )

    take = take.fillna("__MISSING__")

    if axis == "index":
        out = df.reindex(index=take.values)
        if "__MISSING__" in out.index:
            out = out.drop(index="__MISSING__")
    else:
        out = df.reindex(columns=take.values)
        if "__MISSING__" in out.columns:
            out = out.drop(columns="__MISSING__")

    return out


def prepare_interaction_covariate(
    interaction_covariate,
    covariates_df: Optional[pd.DataFrame],
    sample_ids: pd.Index,
) -> Optional[np.ndarray]:
    """
    Normalize and align a sample-level interaction covariate to sample_ids.
    Accepts:
      - str: column name in covariates_df
      - pd.Series with sample IDs in the index
      - 1D array-like matching sample_ids order
    Returns float32 array (n_samples,) centered and scaled (z-score).
    """
    if interaction_covariate is None:
        return None

    if isinstance(interaction_covariate, str):
        if covariates_df is None or interaction_covariate not in covariates_df.columns:
            raise ValueError(
                f"interaction_covariate '{interaction_covariate}' not found in covariates_df."
            )
        series = covariates_df[interaction_covariate]
        series_df = pd.DataFrame({"v": series})
        series_df = align_like_casefold(
            series_df, like=sample_ids, axis="index",
            what="sample IDs in interaction_covariate", strict=True,
        )
        vec = series_df["v"].to_numpy(dtype=np.float64, copy=False)
    elif isinstance(interaction_covariate, pd.Series):
        series_df = pd.DataFrame({"v": interaction_covariate})
        series_df = align_like_casefold(
            series_df, like=sample_ids, axis="index",
            what="sample IDs in interaction_covariate", strict=True,
        )
        vec = series_df["v"].to_numpy(dtype=np.float64, copy=False)
    else:
        vec = np.asarray(interaction_covariate, dtype=np.float64)
        if vec.ndim != 1:
            raise ValueError("interaction_covariate must be 1D.")
        if vec.shape[0] != len(sample_ids):
            raise ValueError(
                f"interaction_covariate length {vec.shape[0]} != n_samples {len(sample_ids)}."
            )

    if not np.all(np.isfinite(vec)):
        raise ValueError("interaction_covariate contains non-finite values.")

    mean = float(vec.mean())
    std = float(vec.std(ddof=0))
    if std <= 0:
        raise ValueError("interaction_covariate has zero variance.")
    vec = (vec - mean) / std
    return vec.astype(np.float32, copy=False)
