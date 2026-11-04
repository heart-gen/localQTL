"""
LD-aware ancestry contrast for the dosage model.

The pairwise contrast of :func:`localqtl.cis.map_nominal`'s dosage model,
``pval_diff_anc{i}_anc{j}``, tests whether the *marginal* ancestry-specific
effects of the tested variant differ.  At a variant that tags the causal
variant instead of being it, the marginal effects differ whenever linkage
disequilibrium between the two differs between ancestries, even when the
causal effect itself is identical in every ancestry.  The contrast at the
joint lead of a permutation scan is therefore anti-conservative as a test of
causal-effect heterogeneity (on the localQTL benchmark's shared-effect design
42 % of the leads had ``pval_diff`` below .05 against 5.3 % at the causal
variant), and a marginal contrast is a descriptive quantity, not evidence
that the regulatory effect differs by ancestry.

:func:`ld_aware_contrast` conditions the lead's ancestry-specific dosages on
the local haplotype background: the pooled genotypes (``condition_on=
"genotype"``) or the ancestry-specific dosages (``"dosage"``) of the other
variants of the phenotype's fine-mapped credible sets (:func:`credible_sets`,
SuSiE on the pooled genotypes of the window).  A shared causal effect carried
by any typed variant of the credible sets is then explained by that variant's
own column and cannot leak into the lead's contrast, while an ancestry
difference of the causal effect still reaches the lead through the
ancestry-specific LD that the pooled column does not carry.  Conditioning on
the pooled genotypes keeps that power; conditioning on the ancestry-specific
dosages removes an ancestry-differential signal of the other variants
altogether and is the stricter form.

Numerically the conditioning block is orthogonalised against the lead's own
pooled genotype and reduced to its rank before it is projected out.  The
contrast ``beta_i - beta_j``, its standard error and the residual degrees of
freedom are invariant to this step (the column space of the design does not
change), and the lead stays identifiable when a conditioning variant is in
perfect LD with it.  The per-ancestry ``slope_ld_anc{k}`` are therefore the
lead's pooled effect split by the conditional contrast, not the coefficients
of a regression that also carries a copy of the lead; read the contrast, not
the individual slopes, as the LD-aware quantity.

The columns are written next to the marginal ones by
:func:`localqtl.cis.map_permutations` (``ld_aware=True``) and
:func:`localqtl.cis.map_conditional_dosage` (``credible_sets_df``):
``slope_ld_anc{k}``, ``slope_se_ld_anc{k}``, ``tstat_ld_anc{k}``,
``pval_ld_anc{k}``, ``fstat_joint_ld``, ``pval_joint_ld``,
``z_diff_ld_anc{i}_anc{j}``, ``pval_diff_ld_anc{i}_anc{j}``,
``n_conditioned_ld`` (conditioning variants), ``rank_ld`` (rank of the
conditioning block after orthogonalisation), ``dof_ld``, and, from the
permutation scan, ``n_cs_ld`` (credible sets of the phenotype), ``cs_size_ld``
(size of the credible set holding the lead, 0 when none does) and ``pip_ld``
(the lead's posterior inclusion probability).
"""

from __future__ import annotations

import numpy as np
import torch

from ..preproc import ancestry_allele_stats
from ..regression_kernels import run_batch_dosage_regression
from .common import ancestry_dosage_design
from .nominal import dosage_result_arrays

__all__ = ["LD_CONDITION_MODES", "SUSIE_DEFAULTS", "ld_columns", "resolve_ld_options",
           "credible_sets", "conditioning_columns", "ld_aware_contrast", "lead_ld_contrast"]

LD_CONDITION_MODES = ("genotype", "dosage")
SUSIE_DEFAULTS = {"L": 10, "coverage": 0.95, "min_abs_corr": 0.5, "max_iter": 200, "tol": 1e-3,
                  "scaled_prior_variance": 0.2}
CS_COLUMNS = ["n_cs_ld", "cs_size_ld", "pip_ld"]


def ld_columns(n_ancestries: int, with_cs: bool = True) -> list[str]:
    """The LD-aware columns for ``n_ancestries`` ancestries (see the module docstring)."""
    cols: list[str] = []
    for k in range(n_ancestries):
        cols.extend([f"slope_ld_anc{k}", f"slope_se_ld_anc{k}", f"tstat_ld_anc{k}", f"pval_ld_anc{k}"])
    cols.extend(["fstat_joint_ld", "pval_joint_ld"])
    for i in range(n_ancestries):
        for j in range(i + 1, n_ancestries):
            cols.extend([f"z_diff_ld_anc{i}_anc{j}", f"pval_diff_ld_anc{i}_anc{j}"])
    cols.extend(["n_conditioned_ld", "rank_ld", "dof_ld"])
    if with_cs:
        cols.extend(CS_COLUMNS)
    return cols


def resolve_ld_options(opts: dict | None) -> dict:
    """``{"condition_on": ..., "susie": {...}}`` from a user dict that may carry either key or the SuSiE
    settings at top level (``L``, ``coverage``, ``min_abs_corr``, ``max_iter``, ``tol``,
    ``scaled_prior_variance``)."""
    opts = dict(opts or {})
    condition_on = str(opts.pop("condition_on", "genotype")).lower()
    if condition_on not in LD_CONDITION_MODES:
        raise ValueError(f"ld_aware condition_on must be one of {LD_CONDITION_MODES}, got {condition_on!r}.")
    susie = dict(SUSIE_DEFAULTS)
    susie.update(opts.pop("susie", {}) or {})
    for key in list(opts):
        if key in SUSIE_DEFAULTS:
            susie[key] = opts.pop(key)
    if opts:
        raise ValueError(f"unknown ld_aware option(s): {sorted(opts)}")
    return {"condition_on": condition_on, "susie": susie}


def _index_array(v) -> np.ndarray:
    """Credible-set member indices as a flat int64 numpy array (SuSiE returns device tensors)."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.int64).reshape(-1)


def _float_array(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64).reshape(-1)


@torch.no_grad()
def credible_sets(y_resid: torch.Tensor, G_resid: torch.Tensor, *, L: int = 10, coverage: float = 0.95,
                  min_abs_corr: float = 0.5, max_iter: int = 200, tol: float = 1e-3,
                  scaled_prior_variance: float = 0.2) -> dict:
    """
    SuSiE on the covariate-residualised pooled genotypes of one phenotype.

    ``y_resid`` is ``(n,)`` and ``G_resid`` ``(m, n)`` (variants x samples), both
    already residualised against the covariates.  Returns ``{"cs": [index
    arrays over the rows of G_resid], "pip": (m,), "converged": bool}``; as in
    :func:`localqtl.susie.map`, credible sets are reported only from a
    converged fit.
    """
    from ..susie import susie

    m = int(G_resid.shape[0])
    if m == 0:
        return {"cs": [], "pip": np.zeros(0, dtype=np.float64), "converged": False}
    X = G_resid.T.contiguous()
    res = susie(X, y_resid.reshape(-1), L=max(1, min(int(L), m)), scaled_prior_variance=scaled_prior_variance,
                coverage=coverage, min_abs_corr=min_abs_corr, max_iter=max_iter, tol=tol)
    sets = res.get("sets") or {}
    cs: list[np.ndarray] = []
    if sets.get("cs") and bool(res.get("converged", False)):
        cs = [_index_array(v) for v in sets["cs"].values()]
    return {"cs": cs, "pip": _float_array(res["pip"]), "converged": bool(res.get("converged", False))}


@torch.no_grad()
def conditioning_columns(G_t: torch.Tensor, H_t: torch.Tensor | None, rez, idx: list[int] | np.ndarray,
                         condition_on: str = "genotype", G_resid: torch.Tensor | None = None) -> torch.Tensor | None:
    """
    ``(n, S)`` covariate-residualised, centred columns of the conditioning variants ``idx`` (rows of
    ``G_t``): their pooled genotypes (``"genotype"``, S = len(idx)) or their ancestry-specific
    dosages (``"dosage"``, S = K * len(idx)).  ``G_resid`` may pass the already residualised
    genotypes.  None when ``idx`` is empty.
    """
    if condition_on not in LD_CONDITION_MODES:
        raise ValueError(f"condition_on must be one of {LD_CONDITION_MODES}, got {condition_on!r}.")
    idx = [int(i) for i in np.asarray(idx, dtype=np.int64).reshape(-1)]
    if not idx:
        return None
    ix_t = torch.as_tensor(idx, dtype=torch.long, device=G_t.device)
    if condition_on == "genotype":
        if G_resid is not None:
            return G_resid.index_select(0, ix_t).T.contiguous()
        rows = G_t.index_select(0, ix_t)
    else:
        if H_t is None:
            raise ValueError("condition_on='dosage' needs the ancestry counts H_t.")
        D_o, _ = ancestry_dosage_design(G_t.index_select(0, ix_t), H_t.index_select(0, ix_t))   # (S, n, K)
        s_, n_, k_ = D_o.shape
        rows = D_o.permute(0, 2, 1).reshape(s_ * k_, n_)
    if rez is not None and getattr(rez, "rank", 0) > 0:
        (rows,) = rez.transform(rows, center=True)
    else:
        rows = rows - rows.mean(1, keepdim=True)
    return rows.T.contiguous()


@torch.no_grad()
def ld_aware_contrast(y_resid: torch.Tensor, D_lead: torch.Tensor, L_lead: torch.Tensor | None,
                      g_lead: torch.Tensor, X_cond: torch.Tensor | None, *, k_eff: int,
                      anc_stats: dict | None = None, min_ancestry_ac: float = 0.0, device: str = "cuda",
                      rank_tol: float = 1e-5) -> dict:
    """
    The dosage model of one variant conditional on a block of background columns.

    ``y_resid`` ``(n,)``, ``D_lead`` ``(1, n, K)`` and ``L_lead`` ``(1, n, q)`` are the
    covariate-residualised phenotype and lead design (:func:`localqtl.cis.common.
    residualize_batch_dosage`), ``g_lead`` ``(n,)`` the lead's residualised pooled genotype and
    ``X_cond`` ``(n, S)`` the residualised conditioning block (:func:`conditioning_columns`; None
    or empty for no conditioning).  The block is orthogonalised against ``g_lead``, reduced to
    its rank (singular values above ``rank_tol`` times the largest column norm of the block) and
    projected out of ``y``,
    ``D`` and ``L``; the regression then runs with ``k_eff + rank`` degrees of freedom consumed.
    Returns the :func:`ld_columns` values without the credible-set bookkeeping.
    """
    y = y_resid.reshape(-1)
    n = int(y.shape[0])
    K = int(D_lead.shape[2])
    rank = 0
    n_cond = 0 if X_cond is None else int(X_cond.shape[1])
    Dk = D_lead[0].T                                              # (K, n)
    Lq = None if L_lead is None else L_lead[0].T                  # (q, n)
    if n_cond:
        X = X_cond.to(y.dtype)
        scale = float(torch.linalg.norm(X, dim=0).max().item())  # the block's scale before orthogonalisation
        g = g_lead.reshape(-1, 1).to(y.dtype)
        gg = float((g * g).sum().item())
        if gg > 0:
            X = X - g @ ((g.T @ X) / gg)                          # orthogonal to the lead's pooled genotype
        X = X - X.mean(0, keepdim=True)
        U, s, _ = torch.linalg.svd(X, full_matrices=False)
        if s.numel() and scale > 0:
            keep = s > scale * float(rank_tol)                    # a column the lead already spans drops out
            rank = int(keep.sum().item())
            if rank:
                U = U[:, keep]                                    # (n, rank)
                y = y - U @ (U.T @ y)
                Dk = Dk - (Dk @ U) @ U.T
                if Lq is not None:
                    Lq = Lq - (Lq @ U) @ U.T
    D2 = Dk.T.unsqueeze(0).contiguous()                           # (1, n, K)
    L2 = None if Lq is None else Lq.T.unsqueeze(0).contiguous()
    res = run_batch_dosage_regression(y=y, D=D2, L=L2, k_eff=int(k_eff) + rank, device=device)
    vals, _, _, _, _, dof = dosage_result_arrays(res, anc_stats=anc_stats, min_ancestry_ac=min_ancestry_ac)
    out: dict[str, float] = {}
    for k in range(K):
        for stat in ("slope", "slope_se", "tstat", "pval"):
            out[f"{stat}_ld_anc{k}"] = float(np.asarray(vals[f"{stat}_anc{k}"]).reshape(-1)[0])
    out["fstat_joint_ld"] = float(np.asarray(vals["fstat_joint"]).reshape(-1)[0])
    out["pval_joint_ld"] = float(np.asarray(vals["pval_joint"]).reshape(-1)[0])
    for i in range(K):
        for j in range(i + 1, K):
            out[f"z_diff_ld_anc{i}_anc{j}"] = float(np.asarray(vals[f"z_diff_anc{i}_anc{j}"]).reshape(-1)[0])
            out[f"pval_diff_ld_anc{i}_anc{j}"] = float(np.asarray(vals[f"pval_diff_anc{i}_anc{j}"]).reshape(-1)[0])
    out["n_conditioned_ld"] = int(n_cond)
    out["rank_ld"] = int(rank)
    out["dof_ld"] = int(dof)
    assert n > 0
    return out


@torch.no_grad()
def lead_ld_contrast(y_resid: torch.Tensor, D_resid: torch.Tensor, L_resid: torch.Tensor | None,
                     G_t: torch.Tensor, H_t: torch.Tensor, rez, ix: int, *, k_eff: int,
                     condition_on: str = "genotype", susie: dict | None = None, cs: dict | None = None,
                     G_resid: torch.Tensor | None = None, min_ancestry_ac: float = 0.0,
                     device: str = "cuda") -> dict:
    """
    The LD-aware contrast of the lead ``ix`` of one phenotype's window.

    ``D_resid`` / ``L_resid`` are the window's residualised dosage design, ``G_t`` / ``H_t`` the
    imputed genotypes and ancestry counts (variants x samples[, K]) and ``rez`` the covariate
    residualiser.  The credible sets come from ``cs`` (:func:`credible_sets`) or are fitted here;
    the conditioning variants are every credible-set member other than the lead.  Returns the
    full :func:`ld_columns` values.
    """
    if G_resid is None:
        if rez is not None and getattr(rez, "rank", 0) > 0:
            (G_resid,) = rez.transform(G_t, center=True)
        else:
            G_resid = G_t - G_t.mean(1, keepdim=True)
    if cs is None:
        cs = credible_sets(y_resid, G_resid, **(susie or {}))
    ix = int(ix)
    sets_ = [_index_array(arr) for arr in cs["cs"]]
    members = sorted({int(i) for arr in sets_ for i in arr} - {ix})
    cs_size = next((int(len(arr)) for arr in sets_ if ix in set(int(i) for i in arr)), 0)
    pip = _float_array(cs["pip"])
    pip_lead = float(pip[ix]) if ix < len(pip) else float("nan")
    X = conditioning_columns(G_t, H_t, rez, members, condition_on=condition_on, G_resid=G_resid)
    one = slice(ix, ix + 1)
    anc_stats = {k: v.detach().cpu().numpy().astype(np.float32)
                 for k, v in ancestry_allele_stats(G_t[one], H_t[one]).items()}
    out = ld_aware_contrast(y_resid, D_resid[one], None if L_resid is None else L_resid[one], G_resid[ix], X,
                            k_eff=k_eff, anc_stats=anc_stats, min_ancestry_ac=min_ancestry_ac, device=device)
    out["n_cs_ld"] = int(len(cs["cs"]))
    out["cs_size_ld"] = int(cs_size)
    out["pip_ld"] = pip_lead
    return out
