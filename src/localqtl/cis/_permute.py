import torch

from ..regression_kernels import (
    run_batch_regression_with_permutations,
    prep_ctx_for_perm,
    perm_chunk_r2,
)

__all__ = [
    "make_perm_ix",
    "roll_for_key",
    "compute_perm_r2_max"
]

@torch.no_grad()
def make_perm_ix(n_samples: int, nperm: int, device: str, seed: int | None) -> torch.Tensor:
    """
    Build a global permutation index tensor of shape (nperm, n_samples) on `device`.
    Reuse this across all phenotypes. Deterministic if seed is set.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return torch.stack(
        [torch.randperm(n_samples, generator=g, device=device) for _ in range(nperm)],
        dim=0
    )


def roll_for_key(perm_ix: torch.Tensor, pid: str | int, seed: int | None) -> torch.Tensor:
    """
    Derive a deterministic row-rotation per phenotype/group to decorrelate
    permutations without regenerating them.
    """
    if seed is None:
        return perm_ix
    shift = abs(hash((seed, str(pid)))) % perm_ix.shape[0]
    return perm_ix.roll(shifts=int(shift), dims=0)


def compute_perm_r2_max(
        y_resid: torch.Tensor, G_resid: torch.Tensor,
        H_resid: torch.Tensor | None, k_eff: int, perm_ix: torch.Tensor,
        device: str, perm_chunk: int, return_nominal: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Chunked permutation scan returning max R² per permutation and optional nominal stats."""
    nperm_total = int(perm_ix.shape[0])
    if nperm_total == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (None, None, None, empty) if return_nominal else (None, None, None, empty)

    y_resid = y_resid.contiguous()
    G_resid = G_resid.contiguous()
    if H_resid is not None:
        H_resid = H_resid.contiguous()
    perm_ix = perm_ix.contiguous()
    
    r2_perm_max = torch.full(
        (nperm_total,), -float("inf"), device=device, dtype=torch.float32,
    )
    n_samples = y_resid.shape[0]
    nominal_b = nominal_s = nominal_t = None

    if H_resid is None:
        if return_nominal:
            nominal_b, nominal_s, nominal_t, _ = run_batch_regression_with_permutations(
                y=y_resid, G=G_resid, H=None, y_perm=None, k_eff=k_eff, device=device
            )

        for off in range(0, nperm_total, perm_chunk):
            sel = perm_ix[off:off + perm_chunk]
            if sel.numel() == 0:
                continue
            chunk = sel.shape[0]
            y_perm = y_resid.index_select(0, sel.reshape(-1)).view(chunk, n_samples).transpose(0, 1)
            betas, ses, tstats, r2_block = run_batch_regression_with_permutations(
                y=y_resid, G=G_resid, H=H_resid, y_perm=y_perm, k_eff=k_eff,
                device=device,
            )
            if nominal_b is None and return_nominal:
                nominal_b, nominal_s, nominal_t = betas, ses, tstats

            r2_perm_max[off:off + chunk] = torch.maximum(
                r2_perm_max[off:off + chunk], r2_block.to(torch.float32),
            )
        if return_nominal:
            return nominal_b, nominal_s, nominal_t, r2_perm_max
        return None, None, None, r2_perm_max
    else:
        ctx, b_nom, s_nom, t_nom = prep_ctx_for_perm(y_resid, G_resid, H_resid, k_eff)
        if return_nominal:
            nominal_b, nominal_s, nominal_t = b_nom, s_nom, t_nom

        for off in range(0, nperm_total, perm_chunk):
            sel = perm_ix[off:off + perm_chunk]
            if sel.numel() == 0:
                continue
            chunk = sel.shape[0]
            y_perm = y_resid.index_select(0, sel.reshape(-1)).view(chunk, n_samples).transpose(0, 1)
            r2_block = perm_chunk_r2(ctx, H_resid, G_resid, y_perm)
            r2_perm_max[off:off + chunk] = torch.maximum(
                r2_perm_max[off:off + chunk], r2_block.to(torch.float32)
            )

        if return_nominal:
            return nominal_b, nominal_s, nominal_t, r2_perm_max
        return None, None, None, r2_perm_max
