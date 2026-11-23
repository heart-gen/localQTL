import os
from contextlib import nullcontext

import torch
from ._torch_utils import move_to_device, resolve_device

try:
    torch.backends.fp32_precision = "ieee"
except Exception:
    pass

__all__ = [
    "Residualizer",
    "resolve_mixed_precision",
    "run_batch_regression",
    "run_batch_interaction_regression",
    "run_batch_regression_with_permutations",
    "perm_chunk_r2", "prep_ctx_for_perm",
    "prep_ctx_for_dosage_perm", "dosage_fstat", "dosage_perm_chunk_fmax", "dosage_perm_chunk_f_at", "dosage_perm_chunk_f_at_and_max",
]

_DEFAULT_MIXED_PRECISION = os.environ.get("LOCALQTL_MIXED_PRECISION", "fp16").lower()


def resolve_mixed_precision(mixed_precision: str | None = None, *, interaction: bool = False) -> str:
    """The CUDA autocast mode of a permutation scan: ``"fp16"`` or ``"off"`` (float32).

    An explicit ``mixed_precision`` is returned as given.  Otherwise the environment variable
    ``LOCALQTL_MIXED_PRECISION`` applies to every scan when it is set, and without it the
    default depends on the scan: ``"off"`` for an interaction scan (``ancestry_model=
    "interaction"`` or an ``interaction_covariate``), ``"fp16"`` for the others (``main``,
    ``dosage``, ``tensorqtl_flavor``).  The interaction scan tests a genotype-by-ancestry (or
    genotype-by-covariate) product after projecting out the genotype and the ancestry it is
    built from; under fp16 that nearly collinear projection loses the precision the partial
    R^2 needs, and the scan stops reproducing the CPU (float32) result.  The mode has no
    effect on CPU.
    """
    if mixed_precision is not None:
        return str(mixed_precision).lower()
    env = os.environ.get("LOCALQTL_MIXED_PRECISION")
    if env:
        return env.lower()
    return "off" if interaction else "fp16"

def _resolve_amp_dtype(device: torch.device, mode: str | None) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if mode is None:
        mode = _DEFAULT_MIXED_PRECISION
    mode = mode.lower()
    if mode not in {"fp16", "off"}:
        mode = "fp16"
    if mode == "off":
        return None
    if torch.cuda.is_available():
        return torch.float16
    return None


def _autocast_ctx(device: torch.device, mode: str | None):
    dtype = _resolve_amp_dtype(device, mode)
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _solve_chol(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve L L^T x = B with dtype-safe casting for AMP."""
    return torch.cholesky_solve(B.to(L.dtype), L)


class Residualizer(object):
    """
    Residualizer for regressing out covariates from genotype/phenotype matrices.
    """
    def __init__(self, C_t: torch.Tensor, tensorqtl_flavor: bool = False):
        self.tensorqtl_flavor = tensorqtl_flavor
        # Center covariates and drop columns with zero variance (e.g., intercept-only)
        n_samples = C_t.shape[0]
        C_centered = C_t - C_t.mean(0)
        norms = torch.linalg.norm(C_centered, dim=0)
        keep = norms > 0

        if keep.any():
            C_use = C_centered[:, keep]
            self.Q_t, _ = torch.linalg.qr(C_use, mode='reduced')
            self.rank = int(self.Q_t.shape[1])
        else:
            # No informative covariates remain; act as identity residualizer.
            self.Q_t = C_t.new_zeros((n_samples, 0))
            self.rank = 0

        # k_eff = degrees of freedom consumed by centring (intercept) plus the
        # projected-out covariates; the kernels compute dof = n - k_eff - p.
        if tensorqtl_flavor:
            self.k_eff = C_t.shape[1] + 1
        else:
            self.k_eff = self.rank + 1
        self.dof = n_samples - self.k_eff - 1   # == tensorQTL's n - 2 - k

        if tensorqtl_flavor and self.rank < C_t.shape[1]:
            print(f"[warning] Covariate matrix has {C_t.shape[1] - self.rank} "
                  "constant or colinear columns. "
                  "tensorQTL flavor uses full column count.")

    def transform(self, *matrices: torch.Tensor, center: bool=True
                  ) -> tuple[torch.Tensor]:
        """
        Residualize one or more matrices in a single GPU pass.
        """
        dev = resolve_device(self.Q_t.device if self.Q_t.numel() else matrices[0].device)
        matrices = tuple(move_to_device(M, dev) for M in matrices)
        if len(matrices) == 1:
            M_t = matrices[0]
            if center:
                M_t = M_t - M_t.mean(1, keepdim=True)
            if self.rank == 0:
                return (M_t,)
            MQ = M_t @ self.Q_t
            return (M_t - MQ @ self.Q_t.T,)

        # Concatenate features along rows
        M_cat = torch.cat(matrices, dim=0)
        if center:
            M_cat = M_cat - M_cat.mean(1, keepdim=True)

        if self.rank == 0:
            out = []
            start = 0
            for M in matrices:
                end = start + M.shape[0]
                out.append(M_cat[start:end])
                start = end
            return tuple(out)

        # Project once with cached Q_t (skinny projection)
        MQ = M_cat @ self.Q_t
        M_cat_resid = M_cat - MQ @ self.Q_t.T

        # Split back into original blocks
        out = []
        start = 0
        for M in matrices:
            end = start + M.shape[0]
            out.append(M_cat_resid[start:end])
            start = end
        return tuple(out)

    def check_orthogonality(self, M_t: torch.Tensor, atol: float = 1e-6) -> float:
        """
        Check maximum absolute correlation between residualized matrix and covariates.
        """
        # Residualize
        M_resid = self.transform(M_t, center=True)[0]

        # Project residuals onto Q
        proj = M_resid @ self.Q_t
        max_corr = proj.abs().max().item()

        if max_corr > atol:
            print(f"Warning: residuals not fully orthogonal (max={max_corr:.2e})")
        return max_corr

    @property
    def P(self):
        raise AttributeError("Residualizer no longer exposes .P; use Q_t with skinny projection.")


def _max_ignore_nan(x: torch.Tensor, dim: int):
    """
    Torch <=1.12 doesn't have torch.nanmax.
    Replace NaNs with -inf then take max; return (values, indices).
    Columns that were all-NaN become -inf; we map those to 0.0.
    """
    x2 = torch.nan_to_num(x, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    vals, idx = torch.max(x2, dim=dim)
    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
    return vals, idx


def run_batch_regression(y, G, H=None, k_eff: int = 0, device="cuda"):
    """
    Batched OLS regression for one phenotype and all variants in a cis-window.

    Parameters
    ----------
    y : torch.Tensor
        (n,) phenotype vector (samples)
    G : torch.Tensor
        (m × n) genotype matrix (variants × samples)
    H : torch.Tensor, optional
        (m × n × (k-1)) haplotype ancestry matrix (variants × samples × ancestries-1)
    k_eff : int, optional
        Number of covariate columns projected out beforehand (effective dof reduction).
    device : str
        "cuda" or "cpu"

    Returns
    -------
    betas : torch.Tensor
        (m × p) regression coefficients (per variant, per predictor)
    ses : torch.Tensor
        (m × p) standard errors
    tstats : torch.Tensor
        (m × p) t-statistics
    """
    device = resolve_device(device)

    y = move_to_device(y, device)
    G = move_to_device(G, device)

    n = y.shape[0]

    # Expand y across variants for batching: (m × n × 1)
    y_exp = y.unsqueeze(0).expand(G.shape[0], -1).unsqueeze(-1)

    # Build design matrix X for each variant
    # G -> (m × n × 1)
    G_exp = G.unsqueeze(-1)

    if H is not None:
        H = move_to_device(H, device)  # (m × n × (k-1))
        X = torch.cat([G_exp, H], dim=2)  # (m × n × p)
    else:
        X = G_exp  # (m × n × 1)

    m, n, p = X.shape

    # Compute XtX and Xty in batch
    XtX = torch.matmul(X.transpose(1, 2), X)      # (m × p × p)
    Xty = torch.matmul(X.transpose(1, 2), y_exp)  # (m × p × 1)

    # Regularize near-singular batches (Tikhonov / ridge with tiny lambda);
    # the absolute floor keeps an all-zero column from making the solve fail.
    diag_scale = torch.diagonal(XtX, dim1=1, dim2=2).abs().mean(dim=1, keepdim=True)  # (m, 1)
    eps = (1e-6 * diag_scale + 1e-8).unsqueeze(-1)  # (m, 1, 1)
    I_p = torch.eye(p, device=device, dtype=XtX.dtype).unsqueeze(0)
    XtX_reg = XtX + eps * I_p

    # Solve for betas
    betas = torch.linalg.solve(XtX_reg, Xty).squeeze(-1)  # (m × p)

    # Residuals and variance estimate
    y_hat = torch.matmul(X, betas.unsqueeze(-1))      # (m × n × 1)
    resid = y_exp - y_hat                             # (m × n × 1)
    dof = max(n - int(k_eff) - p, 1)
    sigma2 = (resid.transpose(1,2) @ resid).squeeze() / dof  # (m,)

    # Standard errors: sqrt(diag(XtX^-1) * sigma2)
    XtX_inv = torch.linalg.inv(XtX_reg)               # (m × p × p)
    var_betas = XtX_inv * sigma2.view(-1,1,1)         # broadcast sigma2
    ses = torch.sqrt(torch.clamp(torch.diagonal(var_betas, dim1=1, dim2=2), min=1e-30))  # (m × p)

    # T-statistics
    tstats = betas / ses

    return betas, ses, tstats


def run_batch_interaction_regression(
    y: torch.Tensor,
    G: torch.Tensor,
    interaction: torch.Tensor,
    k_eff: int = 0,
    device: str = "cuda",
    H: torch.Tensor | None = None,
    gi: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Batched OLS for y ~ g + i + g:i (interaction model).

    All inputs are assumed to be pre-residualized against covariates and centered.
    The product column(s) ``g:i`` should be residualized the same way (pass them
    as ``gi``, shape ``(m, n, ni)``, e.g. from
    :func:`localqtl.cis.common.project_interaction`); when ``gi`` is omitted
    they are built as the raw ``G * interaction`` and the model has no intercept.

    Parameters
    ----------
    y : torch.Tensor
        (n,) phenotype vector (samples).
    G : torch.Tensor
        (m, n) genotype matrix (variants x samples).
    interaction : torch.Tensor
        (n,) or (n, ni) interaction term(s), e.g. ancestry dosage or
        sample-level covariate.
    k_eff : int
        Degrees of freedom already consumed by centring (intercept) and the
        projected-out covariates (``Residualizer.k_eff``); dof = n - k_eff - p.
    device : str
        ``"cuda"`` or ``"cpu"``.
    H : torch.Tensor, optional
        (m, n, pH) per-variant nuisance columns (e.g. local-ancestry counts)
        appended to the design as main effects; they reduce ``dof`` by ``pH``
        and their coefficients are not reported.

    Returns
    -------
    dict
        Keys ``b_g``, ``se_g``, ``tstat_g`` — genotype main effect (m,);
        ``b_i``, ``se_i``, ``tstat_i`` — interaction main effect (m,) or (m, ni);
        ``b_gi``, ``se_gi``, ``tstat_gi`` — genotype x interaction (m,) or (m, ni);
        ``dof`` — scalar degrees of freedom.
    """
    device = resolve_device(device)
    y = move_to_device(y, device)
    G = move_to_device(G, device)
    interaction = move_to_device(interaction, device)

    if interaction.dim() == 1:
        interaction = interaction.unsqueeze(-1)  # (n, 1)

    m, n = G.shape
    ni = interaction.shape[1]
    pH = 0 if H is None else int(H.shape[2])
    nb = 1 + 2 * ni + pH  # g, i_1..i_ni, gi_1..gi_ni, h_1..h_pH

    # Build design matrix X: (m, n, 1+2*ni+pH)
    g_col = G.unsqueeze(-1)                                        # (m, n, 1)
    i_cols = interaction.unsqueeze(0).expand(m, -1, -1)            # (m, n, ni)
    if gi is None:
        gi_cols = G.unsqueeze(-1) * interaction.unsqueeze(0)       # (m, n, ni)
    else:
        gi_cols = move_to_device(gi, device)
        if gi_cols.dim() == 2:
            gi_cols = gi_cols.unsqueeze(-1)

    cols = [g_col, i_cols, gi_cols]
    if pH:
        cols.append(move_to_device(H, device))
    X = torch.cat(cols, dim=2)                                     # (m, n, nb)
    y_exp = y.unsqueeze(0).expand(m, -1).unsqueeze(-1)             # (m, n, 1)

    # Solve normal equations: (X'X) b = X'y
    XtX = X.transpose(1, 2) @ X                                   # (m, nb, nb)
    Xty = X.transpose(1, 2) @ y_exp                               # (m, nb, 1)

    # Gentle regularization for near-singular batches
    I_reg = 1e-6 * torch.eye(nb, device=device, dtype=XtX.dtype).unsqueeze(0)
    XtX = XtX + I_reg

    betas = torch.linalg.solve(XtX, Xty).squeeze(-1)              # (m, nb)

    # Residual variance
    y_hat = (X @ betas.unsqueeze(-1))                              # (m, n, 1)
    resid = y_exp - y_hat
    dof = max(n - int(k_eff) - nb, 1)
    rss = (resid.transpose(1, 2) @ resid).squeeze(-1).squeeze(-1) # (m,)
    sigma2 = rss / dof

    # Standard errors from diagonal of (X'X)^-1 * sigma2
    XtX_inv = torch.linalg.inv(XtX)                               # (m, nb, nb)
    var_diag = torch.diagonal(XtX_inv, dim1=1, dim2=2)            # (m, nb)
    ses = torch.sqrt(torch.clamp(var_diag * sigma2.unsqueeze(-1), min=1e-30))  # (m, nb)

    tstats = betas / ses                                           # (m, nb)

    # Unpack columns: g=0, i=1..ni, gi=ni+1..2*ni (H columns, if any, follow)
    gi_end = 1 + 2 * ni
    result = {
        "b_g":      betas[:, 0],
        "se_g":     ses[:, 0],
        "tstat_g":  tstats[:, 0],
        "b_i":      betas[:, 1:1+ni].squeeze(-1),
        "se_i":     ses[:, 1:1+ni].squeeze(-1),
        "tstat_i":  tstats[:, 1:1+ni].squeeze(-1),
        "b_gi":     betas[:, 1+ni:gi_end].squeeze(-1),
        "se_gi":    ses[:, 1+ni:gi_end].squeeze(-1),
        "tstat_gi": tstats[:, 1+ni:gi_end].squeeze(-1),
        "dof":      dof,
    }
    return result


def run_batch_dosage_regression(
    y: torch.Tensor,
    D: torch.Tensor,
    L: torch.Tensor | None = None,
    k_eff: int = 0,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Batched OLS for the ancestry-specific dosage model
    ``y ~ D_1 + ... + D_K (+ L_1 + ... + L_{K-1})``.

    ``D[:, :, k]`` is the (expected) alternate-allele dosage carried on
    ancestry-``k`` haplotypes, ``sum_k D_k = G``; ``L`` holds optional
    local-ancestry main-effect columns.  All inputs are assumed to be
    pre-residualized against covariates and centered.

    Parameters
    ----------
    y : torch.Tensor
        (n,) phenotype vector (samples).
    D : torch.Tensor
        (m, n, K) ancestry-specific dosages (variants x samples x ancestries).
    L : torch.Tensor, optional
        (m, n, q) local-ancestry covariate columns (typically ``q = K - 1``).
    k_eff : int
        Degrees of freedom already consumed by centring (intercept) and the
        projected-out covariates (``Residualizer.k_eff``); dof = n - k_eff - p.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    dict
        ``betas``, ``ses``, ``tstats`` — (m, K) ancestry-specific effects;
        ``cov`` — (m, K, K) covariance of the ancestry betas;
        ``fstat`` — (m,) joint Wald statistic for H0: beta_1 = ... = beta_K = 0
        (F with ``(K, dof)`` degrees of freedom); ``dof`` — scalar.
    """
    device = resolve_device(device)
    y = move_to_device(y, device)
    D = move_to_device(D, device)
    m, n, K = D.shape
    if L is not None:
        L = move_to_device(L, device)
        X = torch.cat([D, L], dim=2)                               # (m, n, p)
    else:
        X = D
    p = X.shape[2]
    y_exp = y.unsqueeze(0).expand(m, -1).unsqueeze(-1)             # (m, n, 1)

    XtX = X.transpose(1, 2) @ X                                    # (m, p, p)
    Xty = X.transpose(1, 2) @ y_exp                                # (m, p, 1)

    diag_scale = torch.diagonal(XtX, dim1=1, dim2=2).abs().mean(dim=1, keepdim=True)
    eps = 1e-6 * diag_scale.unsqueeze(-1)
    I_p = torch.eye(p, device=device, dtype=XtX.dtype).unsqueeze(0)
    XtX_reg = XtX + eps * I_p

    betas = torch.linalg.solve(XtX_reg, Xty).squeeze(-1)           # (m, p)
    y_hat = X @ betas.unsqueeze(-1)
    resid = y_exp - y_hat
    dof = max(n - int(k_eff) - p, 1)
    rss = (resid.transpose(1, 2) @ resid).squeeze(-1).squeeze(-1)  # (m,)
    sigma2 = rss / dof

    XtX_inv = torch.linalg.inv(XtX_reg)                            # (m, p, p)
    cov_all = XtX_inv * sigma2.view(-1, 1, 1)
    ses = torch.sqrt(torch.clamp(torch.diagonal(cov_all, dim1=1, dim2=2), min=1e-30))
    tstats = betas / ses

    b_anc = betas[:, :K]                                           # (m, K)
    cov_anc = cov_all[:, :K, :K]                                   # (m, K, K)
    # Joint Wald test on the ancestry betas: b' cov^-1 b / K ~ F(K, dof)
    I_k = torch.eye(K, device=device, dtype=cov_anc.dtype).unsqueeze(0)
    cov_reg = cov_anc + 1e-12 * torch.diagonal(cov_anc, dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1) * I_k
    sol = torch.linalg.solve(cov_reg, b_anc.unsqueeze(-1))         # (m, K, 1)
    fstat = (b_anc.unsqueeze(1) @ sol).squeeze(-1).squeeze(-1) / K # (m,)

    return {
        "betas": b_anc,
        "ses": ses[:, :K],
        "tstats": tstats[:, :K],
        "cov": cov_anc,
        "fstat": fstat,
        "dof": dof,
    }


@torch.no_grad()
def prep_ctx_for_dosage_perm(D: torch.Tensor, L: torch.Tensor | None, k_eff: int = 0):
    """
    Precompute the phenotype-independent parts of the ancestry-specific dosage
    model, so that permutations cost a few batched matmuls per chunk.

    The design of variant ``v`` is ``X_v = [D_v, L_v]`` as in
    :func:`run_batch_dosage_regression`; ``(X'X)^-1`` and the inverse of its
    leading ``K x K`` block do not depend on ``y`` and are built once here.

    Parameters
    ----------
    D : torch.Tensor
        (m, n, K) ancestry-specific dosages, residualized and centred.
    L : torch.Tensor, optional
        (m, n, q) local-ancestry main-effect columns.
    k_eff : int
        Degrees of freedom consumed by centring and the projected-out
        covariates; ``dof = n - k_eff - p``.

    Returns
    -------
    dict
        ``X`` (m, n, p), ``A`` = ``(X'X)^-1`` (m, p, p), ``M`` =
        ``inv(A[:, :K, :K])`` (m, K, K), ``dof``, ``K``, ``p``.
    """
    X = D if L is None or L.shape[2] == 0 else torch.cat([D, L], dim=2)
    m, n, p = X.shape
    K = int(D.shape[2])
    XtX = X.transpose(1, 2) @ X
    diag_scale = torch.diagonal(XtX, dim1=1, dim2=2).abs().mean(dim=1, keepdim=True)
    eps = 1e-6 * diag_scale.unsqueeze(-1)
    I_p = torch.eye(p, device=X.device, dtype=XtX.dtype).unsqueeze(0)
    A = torch.linalg.inv(XtX + eps * I_p)
    A_kk = A[:, :K, :K]
    I_k = torch.eye(K, device=X.device, dtype=A.dtype).unsqueeze(0)
    ridge_k = 1e-12 * torch.diagonal(A_kk, dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)
    M = torch.linalg.inv(A_kk + ridge_k * I_k)
    return {"X": X, "A": A, "M": M, "dof": max(n - int(k_eff) - p, 1), "K": K, "p": p}


@torch.no_grad()
def _dosage_fstat_block(ctx: dict, Y: torch.Tensor) -> torch.Tensor:
    """Joint-Wald F of every variant against every column of ``Y`` (n, C) -> (m, C)."""
    X, A, M, K, dof = ctx["X"], ctx["A"], ctx["M"], ctx["K"], ctx["dof"]
    XtY = torch.einsum("mnp,nc->mpc", X, Y)            # (m, p, C)
    B = A @ XtY                                        # (m, p, C)
    yty = (Y * Y).sum(dim=0).unsqueeze(0)              # (1, C)
    rss = (yty - (XtY * B).sum(dim=1)).clamp_min(1e-12)  # (m, C)
    sigma2 = rss / dof
    bK = B[:, :K, :]                                   # (m, K, C)
    quad = torch.einsum("mkc,mkl,mlc->mc", bK, M, bK)  # b' A_KK^-1 b
    return quad / (K * sigma2).clamp_min(1e-30)


@torch.no_grad()
def dosage_fstat(ctx: dict, y: torch.Tensor) -> torch.Tensor:
    """Per-variant joint F statistic (m,) of the dosage model for one phenotype."""
    return _dosage_fstat_block(ctx, y.reshape(-1, 1)).squeeze(-1)


@torch.no_grad()
def dosage_perm_chunk_fmax(ctx: dict, y_perm: torch.Tensor) -> torch.Tensor:
    """
    Maximum joint F across variants for each permuted phenotype (``y_perm``
    ``(n, chunk)``) -> ``(chunk,)``.

    The maximum F is the permutation statistic: for fixed ``(K, dof)`` the
    joint p-value is monotone in F, so the largest F is the smallest p-value
    of the window.

    The sanitisation must match the one the *observed* statistic gets in
    :mod:`localqtl.cis.permutations` (``nan_to_num(..., nan=-1.0)``): a NaN
    must never win the maximum, but an overflowing F must, or a null draw
    more extreme than the observation would be counted as the least extreme
    one and ``pval_perm`` would come out anti-conservative.
    """
    F = _dosage_fstat_block(ctx, y_perm)
    return torch.nan_to_num(F, nan=-1.0).amax(dim=0).to(torch.float32)


@torch.no_grad()
def dosage_perm_chunk_f_at(ctx: dict, y_perm: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
    """
    Joint F of the dosage model at one chosen variant per permuted phenotype
    (``y_perm`` ``(n, chunk)``, ``ix`` ``(chunk,)`` variant indices) -> ``(chunk,)``.

    This is the permutation statistic of the pooled-lead scan
    (``lead_selection="pooled"`` in :mod:`localqtl.cis.permutations`): each
    permutation selects its lead the way the observed scan does, by the pooled
    1-df statistic (:func:`perm_chunk_r2_argmax`), and the joint F is read at
    that variant rather than maximised over the window.  Sanitised like
    :func:`dosage_perm_chunk_fmax`: a NaN F becomes ``-1`` (less extreme than
    any observation), an overflowing F stays infinite (more extreme).
    """
    F = _dosage_fstat_block(ctx, y_perm)                                  # (m, chunk)
    ix = ix.to(device=F.device, dtype=torch.long).view(1, -1)
    return torch.nan_to_num(F.gather(0, ix).squeeze(0), nan=-1.0).to(torch.float32)


@torch.no_grad()
def dosage_perm_chunk_f_at_and_max(ctx: dict, y_perm: torch.Tensor, ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :func:`dosage_perm_chunk_f_at` and :func:`dosage_perm_chunk_fmax` from one F
    block: ``(f_at, f_max)``, ``(chunk,)`` each.  The pooled-lead scan needs both
    nulls (0.4.5): the F at the permutation's pooled lead calibrates the
    phenotype-level p-value, the maximum over the window is the null of the
    window-minimum p-value from which the pair extraction reads its nominal
    threshold.  The block is computed once; the second statistic is a reduction.
    """
    F = torch.nan_to_num(_dosage_fstat_block(ctx, y_perm), nan=-1.0)       # (m, chunk)
    ix = ix.to(device=F.device, dtype=torch.long).view(1, -1)
    return F.gather(0, ix).squeeze(0).to(torch.float32), F.amax(dim=0).to(torch.float32)


def _is_cuda(*tensors):
    return all(getattr(t, "is_cuda", False) for t in tensors if t is not None)


def _slope_se(y_ss: torch.Tensor, g_ss: torch.Tensor, r: torch.Tensor, dof: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Standard error of a single-predictor slope from its (partial) correlation:
    ``sqrt(RSS / dof / g_ss)`` with ``RSS = y_ss * (1 - r^2)``, where ``y_ss`` and
    ``g_ss`` are the sums of squares of the phenotype and the predictor after
    everything else is projected out.  Algebraically ``|beta| / |t|``, but
    computed from the sums of squares so that it does not depend on the sign of
    the effect and stays positive at ``beta == 0``.
    """
    one_minus_r2 = torch.clamp(1.0 - r.float() * r.float(), min=eps)
    return torch.sqrt(y_ss.float() * one_minus_r2 / (float(dof) * g_ss.float()))


@torch.no_grad()
def run_batch_regression_with_permutations(
        y: torch.Tensor, G: torch.Tensor, H: torch.Tensor | None = None,
        y_perm: torch.Tensor | None = None, k_eff: int = 0, device: str = "cuda",
        mixed_precision: str | None = None,
):
    """
    Return nominal betas/ses/tstats for G~y, and per-permutation max r^2 across variants.
    No (chunk x chunk) allocations; memory scales with (m*n + (m+n)*chunk).
    """
    device = resolve_device(device)
    y = move_to_device(y, device)
    G = move_to_device(G, device)
    if y_perm is not None:
        y_perm = move_to_device(y_perm, device)

    EPS = 1e-8
    n = y.shape[0]
    m = G.shape[0]

    if H is None:
        # Fast vectorized implementation (no H covariates)
        dof = max(n - 1 - int(k_eff), 1)
        with _autocast_ctx(device, mixed_precision) if _is_cuda(G, y) else nullcontext():
            Gnorm2 = (G * G).sum(dim=1) + EPS          # (m,)
            ynorm2 = (y * y).sum() + EPS               # scalar
            Gy     = G @ y                             # (m,)

        r      = Gy / torch.sqrt(Gnorm2 * ynorm2)  # (m,)
        one_minus_r2 = 1.0 - r * r
        t      = r * torch.sqrt(dof / torch.clamp(one_minus_r2, min=EPS))
        beta   = Gy / Gnorm2                         # (m,)
        se     = _slope_se(ynorm2, Gnorm2, r, dof, EPS)

        betas  = beta.unsqueeze(1)                   # (m,1)
        ses    = se.unsqueeze(1)                     # (m,1)
        tstats = t.unsqueeze(1)                      # (m,1)

        # Permutations (chunked)
        r2_perm = None
        if y_perm is not None:
            Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS
            if _is_cuda(G, y_perm):
                with _autocast_ctx(device, mixed_precision):
                    GYp = G @ y_perm             # (m, chunk)
            else:
                GYp = G @ y_perm

            GYp = GYp.float()
            denom = torch.sqrt(Gnorm2).unsqueeze(1) * torch.sqrt(Ypnorm2).unsqueeze(0)
            R2 = (GYp / torch.clamp(denom, min=EPS)) ** 2
            r2_vals, _ = _max_ignore_nan(R2, dim=0)
            r2_perm = r2_vals.float()

        return betas, ses, tstats, r2_perm
    else:
        # Nuisance columns H: project them out of the tested column once and take
        # every permutation from one matrix product (prep_ctx_for_perm).
        H = move_to_device(H, device)
        assert H.dim() == 3, "H must be (m, n, pH)"
        ctx, betas, ses, tstats = prep_ctx_for_perm(y, G, H, k_eff, mixed_precision=mixed_precision)
        r2_perm = None
        if y_perm is not None:
            y_perm = y_perm.to(device, non_blocking=True)        # (n, chunk)
            r2_perm = perm_chunk_r2(ctx, H, G, y_perm, mixed_precision=mixed_precision)
        return betas, ses, tstats, r2_perm


# A tested column whose squared norm, after the nuisance columns are projected out, is below this
# fraction of its squared norm before is collinear with them at float32 resolution (in the
# interaction model: g*h at a variant whose carriers all have the same local ancestry, so that
# g*h = c*g).  It carries no test: its nominal statistics are NaN and its null R^2 is 0.
_COLLINEAR_TOL = 1e-9


@torch.no_grad()
def prep_ctx_for_perm(y: torch.Tensor, G: torch.Tensor, H: torch.Tensor, k_eff: int,
                      mixed_precision: str | None = None):
    """
    Everything the permutations of one phenotype reuse when nuisance columns ``H`` are present,
    plus the nominal statistics.

    Per variant, with ``S = H'H = L L'``: the nuisance columns are whitened once,
    ``W = L^-1 H'`` (``W W' = I``), and each tested column has them projected out and is scaled
    to unit norm, ``x~ = (x - W'W x) / |x - W'W x|`` (two passes, so that ``x~`` is orthogonal
    to ``W`` at float32 resolution even when ``x`` is nearly collinear with ``H``, as a
    genotype-by-ancestry product is with the genotype and the ancestry).  The rows of ``x~`` and
    ``W`` are stacked in ``ctx["Z"]``, and the partial R^2 of ``x`` given ``H`` for a phenotype
    ``v`` is ``(x~.v)^2 / (v.v - |W v|^2)``: one product ``Z @ v`` per chunk of permuted
    phenotypes (:func:`perm_chunk_r2`), with no per-chunk solve.  The projection is done here
    in float32; only that product runs under autocast.

    Shapes: ``y (n,)``, ``H (m, n, pH)``, ``G (m, n)`` for one tested column or ``(m, n, q)``
    for ``q`` columns tested one at a time against the same ``H`` (the ancestries of the
    interaction model).  Returns ``(ctx, betas, ses, tstats)``, each statistic ``(m, 1)`` or
    ``(m, q)``; ``mixed_precision`` is accepted for the callers and not used here.
    """
    device = y.device
    m, n, pH = H.shape
    EPS = 1e-8
    dof = max(n - (1 + pH) - int(k_eff), 1)
    X = G if G.dim() == 3 else G.unsqueeze(-1)                    # (m, n, q)
    q = int(X.shape[2])
    X = X.to(torch.float32).transpose(1, 2)                       # (m, q, n)
    Ht = H.to(torch.float32).transpose(1, 2)                      # (m, pH, n)
    y = y.to(torch.float32)

    S = Ht @ Ht.transpose(1, 2)                                   # (m, pH, pH)
    L, info = torch.linalg.cholesky_ex(S)
    if (info != 0).any():
        I = torch.eye(pH, device=device, dtype=torch.float32).expand_as(S)
        L = torch.linalg.cholesky(S + (1e-6 * I))
    W = torch.linalg.solve_triangular(L, Ht, upper=False)         # (m, pH, n)
    Wt = W.transpose(1, 2)

    x_ss = (X * X).sum(dim=2)                                     # (m, q)
    Xo = X - (X @ Wt) @ W
    Xo = Xo - (Xo @ Wt) @ W
    G_resid2 = (Xo * Xo).sum(dim=2)                               # (m, q)
    tested = G_resid2 > _COLLINEAR_TOL * x_ss
    scale = torch.where(tested, torch.rsqrt(torch.clamp(G_resid2, min=1e-30)), torch.zeros_like(G_resid2))
    Xu = Xo * scale.unsqueeze(-1)                                 # unit rows; zero where not tested
    Z = torch.cat([Xu, W], dim=1).reshape(m * (q + pH), n).contiguous()

    # nominal statistics from the same design as the null
    A = (Z @ y).view(m, q + pH)
    y_resid2 = ((y * y).sum() - (A[:, q:] * A[:, q:]).sum(dim=1)).clamp_min(EPS)   # (m,)
    r = A[:, :q] / torch.sqrt(y_resid2).unsqueeze(1)
    r = torch.clamp(r, min=-1 + 1e-7, max=1 - 1e-7)
    t = r * torch.sqrt(float(dof) / torch.clamp(1.0 - r * r, min=EPS))
    nan = torch.full_like(r, float("nan"))
    betas = torch.where(tested, A[:, :q] * scale, nan)
    ses = torch.where(tested, _slope_se(y_resid2.unsqueeze(1), torch.clamp(G_resid2, min=1e-30), r, dof, EPS), nan)
    tstats = torch.where(tested, t, nan)

    ctx = {"Z": Z, "m": m, "q": q, "pH": pH, "L": L, "G_resid2": G_resid2 if G.dim() == 3 else G_resid2[:, 0],
           "tested": tested if G.dim() == 3 else tested[:, 0], "dof": dof}
    return ctx, betas, ses, tstats


@torch.no_grad()
def _perm_chunk_r2_block(ctx: dict, H: torch.Tensor, G: torch.Tensor, y_perm: torch.Tensor,
                         mixed_precision: str | None = None) -> torch.Tensor:
    """
    Partial R^2 of every variant against every permuted phenotype, ``(n, chunk) -> (m, chunk)``,
    NaN as 0; with several tested columns, the largest over the columns.  ``H`` and ``G`` are
    the arrays ``ctx`` was built from (:func:`prep_ctx_for_perm`) and are not read again.
    """
    EPS = 1e-8
    Z, m, q, pH = ctx["Z"], ctx["m"], ctx["q"], ctx["pH"]
    y_perm = y_perm.to(Z.dtype)
    with (_autocast_ctx(y_perm.device, mixed_precision) if _is_cuda(Z, y_perm) else nullcontext()):
        A = Z @ y_perm                                            # (m * (q + pH), chunk)
    A = A.to(torch.float32).view(m, q + pH, -1)
    A = A * A
    Ypnorm2 = (y_perm * y_perm).sum(dim=0) + EPS                  # (chunk,)
    Yp_resid2 = (Ypnorm2.unsqueeze(0) - A[:, q:].sum(dim=1)) + EPS
    num = A[:, 0] if q == 1 else A[:, :q].amax(dim=1)
    return torch.nan_to_num(num / Yp_resid2, nan=0.0)


@torch.no_grad()
def perm_chunk_r2(ctx: dict, H: torch.Tensor, G: torch.Tensor, y_perm: torch.Tensor,
                  mixed_precision: str | None = None) -> torch.Tensor:
    """
    Compute max R^2 across variants for a permutation chunk (n x chunk),
    using precomputed ctx from prep_ctx_for_perm.
    Returns (chunk,) on the same device.
    """
    return _perm_chunk_r2_block(ctx, H, G, y_perm, mixed_precision).amax(dim=0).to(torch.float32)


@torch.no_grad()
def perm_chunk_r2_argmax(ctx: dict, H: torch.Tensor, G: torch.Tensor, y_perm: torch.Tensor,
                         mixed_precision: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The pooled lead of every permuted phenotype: the maximum partial R^2
    across variants and the index of the variant that attains it, ``(chunk,)``
    each, from the ctx of :func:`prep_ctx_for_perm`.  The statistic is the one
    :func:`perm_chunk_r2` maximises, so the selection made under the null is
    the selection the observed pooled scan makes.
    """
    r2_max, ix = _perm_chunk_r2_block(ctx, H, G, y_perm, mixed_precision).max(dim=0)
    return r2_max.to(torch.float32), ix


if __name__ == "__main__" and os.environ.get("LOCALQTL_DEV_CHECK", "0") == "1":
    torch.manual_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n, pH = 64, 128, 2
    y = torch.randn(n, device=device, dtype=torch.float32)
    G = torch.randn(m, n, device=device, dtype=torch.float32)
    H = torch.randn(m, n, pH, device=device, dtype=torch.float32)
    y_perm = torch.randn(n, 3, device=device, dtype=torch.float32)

    ctx_amp, betas_amp, ses_amp, tstats_amp = prep_ctx_for_perm(
        y, G, H, k_eff=0, mixed_precision="fp16"
    )
    ctx_fp32, betas_fp32, ses_fp32, tstats_fp32 = prep_ctx_for_perm(
        y, G, H, k_eff=0, mixed_precision="off"
    )

    max_beta = (betas_amp - betas_fp32).abs().max().item()
    max_se = (ses_amp - ses_fp32).abs().max().item()
    max_t = (tstats_amp - tstats_fp32).abs().max().item()
    worst = max(max_beta, max_se, max_t)
    assert worst <= 1e-5, f"stat mismatch under AMP ({worst:.2e})"

    perm_vals = perm_chunk_r2(ctx_amp, H, G, y_perm, mixed_precision="fp16")
    assert torch.isfinite(perm_vals).all(), "Permutation path produced NaNs/inf"
