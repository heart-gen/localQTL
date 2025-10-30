import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from ..regression_kernels import Residualizer

def dosage_vector_for_covariate(genotype_df: pd.DataFrame, variant_id: str,
                                sample_order: pd.Index, missing: float | int | None) -> np.ndarray:
    """Fetch a dosage row aligned to samples; impute 'missing' to mean of observed."""
    if not genotype_df.columns.equals(sample_order):
        row = genotype_df.loc[variant_id, sample_order].to_numpy(dtype=np.float32, copy=True)
    else:
        row = genotype_df.loc[variant_id].to_numpy(dtype=np.float32, copy=True)
    if missing is not None:
        mm = (row == missing)
        if mm.any():
            if (~mm).any():
                row[mm] = row[~mm].mean(dtype=np.float32)
            else:
                row[mm] = 0.0
    return row


def residualize_matrix_with_covariates(
        Y: torch.Tensor, C: Optional[pd.DataFrame], device: str
) -> Tuple[torch.Tensor, Optional[Residualizer]]:
    """
    Residualize (features x samples) matrix Y against covariates C across samples.
    Returns residualized Y and a Residualizer (or None if no covariates).
    """
    if C is None:
        return Y, None
    C_t = torch.tensor(C.values, dtype=torch.float32, device=device)
    rez = Residualizer(C_t)
    (Y_resid,) = rez.transform(Y, center=True)
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
    dev = (rez.Q_t.device if (rez is not None and hasattr(rez, "Q_t")) else G.device)
    G = G.to(dev)
    if H is not None:
        H = H.to(dev)

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
            Y = torch.stack([yi if torch.is_tensor(yi) else torch.as_tensor(yi, dtype=torch.float32, device=dev)
                             for yi in y], dim=0).to(dev)
        else:
            Y = (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32, device=dev))
            if Y.ndim == 1:
                Y = Y.unsqueeze(0)
            Y = Y.to(dev)
    else:
        Y = (y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32, device=dev))
        if Y.ndim == 1:
            Y = Y.unsqueeze(0)
        Y = Y.to(dev)
    
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
