import torch
import numpy as np

__all__ = [
    "get_pairwise_ld",
    "get_ld_matrix",
]

def get_pairwise_ld(pgr, id1, id2=None, r2=True, use_torch=False, dtype=np.float32):
    """
    Compute LD (r or r²) between one or more variants.

    Parameters
    ----------
    pgr : PgenReader
        Initialized PgenReader object.
    id1 : str or list
        Variant ID(s).
    id2 : str or list, optional
        Second set of variant ID(s). If None, computes LD within id1.
    r2 : bool
        If True, return r² (squared correlation). Otherwise return r.
    use_torch : bool
        If True, use torch on GPU if available.
    dtype : np.float32 or np.float64
        Precision of computation.

    Returns
    -------
    ld : float, np.ndarray
        LD value(s). Shape depends on input.
    """
    if isinstance(id1, str):
        id1 = [id1]
    if id2 is None:
        id2 = id1 if len(id1) > 1 else id1
    elif isinstance(id2, str):
        id2 = [id2]

    g1 = pgr.read_list(id1, dosages=False, to_torch=use_torch)
    g2 = pgr.read_list(id2, dosages=False, to_torch=use_torch)

    if use_torch:
        g1 = g1.to(dtype) - g1.mean(dim=1, keepdim=True)
        g2 = g2.to(dtype) - g2.mean(dim=1, keepdim=True)
    else:
        g1 = g1.astype(dtype, copy=False) - g1.mean(1, keepdims=True)
        g2 = g2.astype(dtype, copy=False) - g2.mean(1, keepdims=True)

    # Case 1: both single SNPs
    if len(id1) == 1 and len(id2) == 1:
        if use_torch:
            num = (g1 * g2).sum()
            denom = torch.sqrt((g1**2).sum() * (g2**2).sum())
            r = num / denom
            return (r**2).item() if r2 else r.item()
        else:
            num = (g1 * g2).sum()
            denom = np.sqrt((g1**2).sum() * (g2**2).sum())
            r = num / denom
            return (r**2) if r2 else r

    # Case 2: one vs many
    if len(id1) == 1 or len(id2) == 1:
        if use_torch:
            num = (g1 * g2).sum(1)
            denom = torch.sqrt((g1**2).sum() * (g2**2).sum(1))
            r = num / denom
            return (r**2).cpu().numpy() if r2 else r.cpu().numpy()
        else:
            num = (g1 * g2).sum(1)
            denom = np.sqrt((g1**2).sum() * (g2**2).sum(1))
            r = num / denom
            return (r**2) if r2 else r

    # Case 3: many vs many → matrix
    if use_torch:
        num = torch.matmul(g1, g2.T)
        denom = torch.sqrt((g1**2).sum(1, keepdim=True) * (g2**2).sum(1))
        r = num / denom
        return (r**2).cpu().numpy() if r2 else r.cpu().numpy()
    else:
        num = np.dot(g1, g2.T)
        denom = np.sqrt((g1**2).sum(1)[:, None] * (g2**2).sum(1))
        r = num / denom
        return (r**2) if r2 else r


def get_ld_matrix(pgr, variant_ids=None, anchor_variant=None, window=None,
                  r2=True, use_torch=False, dtype=np.float32):
    """
    Compute a full LD matrix for a set of variants or a cis-window.

    Parameters
    ----------
    pgr : PgenReader
        Initialized PgenReader object.
    variant_ids : list, optional
        Explicit list of variant IDs to compute LD for.
    anchor_variant : str, optional
        Variant ID used as anchor for cis-window.
    window : int, optional
        Window size in base pairs (+/-) around anchor_variant.
    r2 : bool
        If True, return r² matrix. Otherwise return r matrix.
    use_torch : bool
        If True, compute with torch on GPU if available.
    dtype : np.float32 or np.float64
        Precision of computation.

    Returns
    -------
    ld_matrix : np.ndarray
        LD matrix (n_variants x n_variants).
    var_ids : list
        Variant IDs in the same order as the LD matrix.
    """
    if variant_ids is None:
        if anchor_variant is None or window is None:
            raise ValueError("Must provide either variant_ids or (anchor_variant + window).")

        # Locate anchor variant
        if anchor_variant not in pgr.variant_df.index:
            raise ValueError(f"Anchor variant {anchor_variant} not found in PgenReader.variant_df")

        chrom = pgr.variant_df.loc[anchor_variant, "chrom"]
        pos = pgr.variant_df.loc[anchor_variant, "pos"]

        start = pos - window
        end = pos + window
        mask = (pgr.variant_df["chrom"] == chrom) & \
               (pgr.variant_df["pos"].between(start, end))
        variant_ids = pgr.variant_df.index[mask].tolist()

    # Load genotype block
    G = pgr.read_list(variant_ids, dosages=False, to_torch=use_torch)

    if use_torch:
        G = G.to(dtype)
        G = G - G.mean(dim=1, keepdim=True)
        denom = torch.sqrt((G**2).sum(1, keepdim=True))
        G = G / denom
        corr = torch.matmul(G, G.T)
        ld = corr**2 if r2 else corr
        return ld.cpu().numpy(), variant_ids
    else:
        G = G.astype(dtype, copy=False)
        G = G - G.mean(1, keepdims=True)
        denom = np.sqrt((G**2).sum(1))[:, None]
        G = G / denom
        corr = np.dot(G, G.T)
        ld = corr**2 if r2 else corr
        return ld, variant_ids
