"""
Ancestry-resolved effects at a fixed set of signals.

:func:`map_conditional_dosage` does not search a cis-window: it takes signals
that were already selected (typically the output of
:func:`localqtl.cis.map_independent` or the lead rows of
:func:`localqtl.cis.map_permutations`) and re-fits each of them with the
ancestry-specific dosage model, conditioning on the other signals of the same
phenotype.  Because selection has already happened there is no multiple-testing
correction to redo, so no permutations are needed and the result is one row per
signal with the per-ancestry effects, the joint test and the pairwise
difference tests of :func:`localqtl.cis.map_nominal`.
"""

import numpy as np
import pandas as pd
import torch

from ..haplotypeio import InputGeneratorCisWithHaps
from ..preproc import allele_stats, ancestry_allele_stats, impute_mean_and_filter
from ..regression_kernels import Residualizer, run_batch_dosage_regression
from ..utils import SimpleLogger
from .common import ancestry_dosage_design, residualize_batch_dosage
from .nominal import _dosage_columns, _n_ancestries, dosage_result_arrays

__all__ = ["map_conditional_dosage"]

CONDITION_MODES = ("genotype", "dosage", "none")


def _signal_sets(signals_df: pd.DataFrame, phenotype_df: pd.DataFrame,
                 genotype_df: pd.DataFrame) -> dict[str, list[str]]:
    """Validate ``signals_df`` and return the ordered variant list per phenotype."""
    for col in ("phenotype_id", "variant_id"):
        if col not in signals_df.columns:
            raise ValueError(f"signals_df must contain '{col}'.")
    unknown_p = set(signals_df["phenotype_id"]) - set(phenotype_df.index)
    if unknown_p:
        raise ValueError(f"{len(unknown_p)} signal phenotypes are not in phenotype_df "
                         f"(first: {sorted(unknown_p)[:5]})")
    unknown_v = set(signals_df["variant_id"]) - set(genotype_df.index)
    if unknown_v:
        raise ValueError(f"{len(unknown_v)} signal variants are not in genotype_df "
                         f"(first: {sorted(unknown_v)[:5]})")
    sets: dict[str, list[str]] = {}
    for pid, sub in signals_df.groupby("phenotype_id", sort=False):
        if "rank" in sub.columns:
            sub = sub.sort_values("rank")
        seen: list[str] = []
        for vid in sub["variant_id"]:
            if vid not in seen:
                seen.append(vid)
        sets[str(pid)] = seen
    return sets


def map_conditional_dosage(
        genotype_df: pd.DataFrame,
        variant_df: pd.DataFrame,
        phenotype_df: pd.DataFrame,
        phenotype_pos_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        haplotypes: object,
        covariates_df: pd.DataFrame | None = None,
        loci_df: pd.DataFrame | None = None,
        *,
        condition_on: str = "genotype",
        min_ancestry_ac: float = 0.0,
        device: str = "cuda",
        tensorqtl_flavor: bool = False,
        locus_method: str = "stepwise",
        nearest_tolerance: int | None = None,
        logger: SimpleLogger | None = None,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Ancestry-specific effects at already-selected signals.

    Each signal is fitted as ``y ~ D_1..D_K + L + covariates [+ the other
    signals of that phenotype]``, where ``D_k = g * l_k / 2`` is the dosage
    carried on ancestry-``k`` haplotypes and ``L`` are the local-ancestry main
    effects.  Only the listed variants are read, so this is cheap enough to run
    on a prioritised locus list.

    Parameters
    ----------
    signals_df : DataFrame
        Needs ``phenotype_id`` and ``variant_id``; ``rank`` (as produced by
        :func:`localqtl.cis.map_independent`) orders the signals of a
        phenotype and is carried through to the output.  Any further columns
        are ignored.
    haplotypes
        Local ancestry: a :class:`localqtl.haplotypeio.LocalAncestryPanel` or a
        ``(loci, samples, channels)`` array with ``loci_df``, as for
        :func:`localqtl.cis.map_nominal`.
    condition_on : {"genotype", "dosage", "none"}
        What to project out for the *other* signals of the same phenotype.
        ``"genotype"`` (default) uses their pooled dosage, which matches how
        :func:`localqtl.cis.map_independent` defined them and costs one
        degree of freedom each.  ``"dosage"`` uses their K ancestry-specific
        dosages instead, so an ancestry-differential secondary signal cannot
        leak into the variant being tested; it costs K degrees of freedom per
        conditioned signal.  ``"none"`` fits every signal marginally.
    min_ancestry_ac : float, default 0
        Ancestries whose in-sample minor-allele count is below this value get
        NaN effect estimates and pairwise differences (the joint test and the
        allele statistics are unchanged), as in
        :func:`localqtl.cis.map_nominal`.

    Returns
    -------
    DataFrame
        One row per signal: ``phenotype_id``, ``variant_id``, ``rank`` (when
        given), ``n_conditioned``, ``start_distance``, ``end_distance``,
        ``af``, ``ma_samples``, ``ma_count``, the lead-ancestry summary
        (``slope``, ``slope_se``, ``tstat``, ``pval_nominal``) and the
        per-ancestry columns of the dosage model.

    Notes
    -----
    The p-values are conditional nominal p-values at variants that were chosen
    on the same data, so they are not corrected for the original selection.
    Read them as "how different are the ancestry-specific effects of this
    established signal", not as evidence that the signal exists.
    """
    condition_on = str(condition_on).lower()
    if condition_on not in CONDITION_MODES:
        raise ValueError(f"condition_on must be one of {CONDITION_MODES}, got {condition_on!r}.")
    if haplotypes is None:
        raise ValueError("map_conditional_dosage requires haplotypes (the dosage model needs local ancestry).")
    device = device if device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or SimpleLogger(verbose=verbose, timestamps=True)

    sets = _signal_sets(signals_df, phenotype_df, genotype_df)
    if not sets:
        raise ValueError("signals_df is empty.")
    ranks = {}
    if "rank" in signals_df.columns:
        ranks = {(str(r.phenotype_id), str(r.variant_id)): r.rank
                 for r in signals_df.itertuples()}

    # The generator bisects variant_df['pos'] within each chromosome, so the
    # sub-frames have to be grouped by chromosome and ascending in position.
    # Ordering them by variant ID instead silently drops the signals of any
    # phenotype whose IDs do not happen to sort the way its positions do.
    vsub = variant_df.loc[sorted({v for vs in sets.values() for v in vs})]
    order = np.lexsort((vsub["pos"].to_numpy(dtype=np.int64),
                        vsub["chrom"].astype(str).to_numpy()))
    needed = list(vsub.index[order])
    vsub = vsub.iloc[order]
    gsub = genotype_df.loc[needed]

    # One "window" per phenotype, spanning exactly its own signals: window=0
    # makes the generator serve the closed interval [start, end].
    rows_pos = {}
    for pid, vs in sets.items():
        pos = vsub.loc[vs, "pos"].to_numpy(dtype=np.int64)
        chroms = set(vsub.loc[vs, "chrom"].astype(str))
        if len(chroms) > 1:
            raise ValueError(f"signals of {pid} span several chromosomes: {sorted(chroms)}")
        rows_pos[pid] = (chroms.pop(), int(pos.min()), int(pos.max()))
    pos_sub = pd.DataFrame(
        {"chr": [rows_pos[p][0] for p in sets],
         "start": [rows_pos[p][1] for p in sets],
         "end": [rows_pos[p][2] for p in sets]},
        index=list(sets),
    )
    phen_sub = phenotype_df.loc[list(sets)]

    # ``pos_sub`` exists only to make the generator serve each phenotype's own
    # signals; the reported distances are measured against the phenotype's
    # real start/end so that they line up with the other cis outputs.
    real_pos = phenotype_pos_df.loc[list(sets)]
    if "pos" in real_pos.columns:
        real_start = real_pos["pos"].astype(np.int64).to_dict()
        real_end = dict(real_start)
    else:
        real_start = real_pos["start"].astype(np.int64).to_dict()
        real_end = real_pos["end"].astype(np.int64).to_dict()

    logger.write(f"localQTL: conditional ancestry-specific effects at {len(needed)} variants "
                 f"across {len(sets)} phenotypes")
    logger.write(f"  * conditioning on other signals: {condition_on}")
    if min_ancestry_ac and min_ancestry_ac > 0:
        logger.write(f"  * masking ancestry effects with in-sample minor-allele count < {min_ancestry_ac:g}")

    cov_t = None
    if covariates_df is not None:
        cov_t = torch.as_tensor(covariates_df.loc[list(phenotype_df.columns)].to_numpy(np.float32),
                                dtype=torch.float32, device=device)

    ig = InputGeneratorCisWithHaps(
        genotype_df=gsub, variant_df=vsub, phenotype_df=phen_sub,
        phenotype_pos_df=pos_sub, window=0, haplotypes=haplotypes, loci_df=loci_df,
        locus_method=locus_method, nearest_tolerance=nearest_tolerance,
    )
    idx_to_id = vsub.index.to_numpy()
    n_anc_out = _n_ancestries(haplotypes, "dosage")

    # The design is the covariate matrix alone whenever nothing is projected
    # out (condition_on='none', or a phenotype with a single signal), so its
    # QR is built once rather than once per signal.
    n_samples = int(phenotype_df.shape[1])
    base_design = (cov_t if cov_t is not None
                   else torch.zeros((n_samples, 0), dtype=torch.float32, device=device))
    base_rez: Residualizer | None = None

    records: list[dict] = []
    for batch in ig.generate_data():
        # Built without group_s, so every batch is one phenotype.
        p, G_block, v_idx, H_block, pid = batch

        y_t = torch.as_tensor(np.asarray(p, dtype=np.float32), dtype=torch.float32, device=device)
        G_t = torch.as_tensor(np.asarray(G_block, dtype=np.float32), dtype=torch.float32, device=device)
        H_t = torch.as_tensor(np.asarray(H_block, dtype=np.float32), dtype=torch.float32, device=device)
        G_t, keep, _ = impute_mean_and_filter(G_t)
        keep_np = keep.detach().cpu().numpy()
        v_idx = v_idx[keep_np]
        H_t = H_t[keep]

        ids_here = [str(idx_to_id[i]) for i in v_idx]
        where = {vid: i for i, vid in enumerate(ids_here)}
        wanted = [v for v in sets[str(pid)] if v in where]
        dropped = [v for v in sets[str(pid)] if v not in where]
        if dropped:
            logger.write(f"    ** {pid}: dropping {len(dropped)} monomorphic or unmapped "
                         f"signal(s): {dropped[:5]}")
        if not wanted:
            continue

        af_t, ma_samples_t, ma_count_t = allele_stats(G_t, ploidy=2)
        start_pos = real_start[str(pid)]
        end_pos = real_end[str(pid)]

        for vid in wanted:
            t = where[vid]
            others = [where[o] for o in wanted if o != vid] if condition_on != "none" else []
            if others:
                extra = [] if cov_t is None else [cov_t]
                ix = torch.as_tensor(others, dtype=torch.long, device=device)
                if condition_on == "genotype":
                    extra.append(G_t.index_select(0, ix).T.contiguous())       # (n, S-1)
                else:
                    D_o, _ = ancestry_dosage_design(G_t.index_select(0, ix),
                                                    H_t.index_select(0, ix))   # (S-1, n, K)
                    extra.append(D_o.permute(1, 0, 2).reshape(int(G_t.shape[1]), -1))
                rez = Residualizer(torch.cat(extra, dim=1), tensorqtl_flavor=tensorqtl_flavor)
            else:
                if base_rez is None:
                    base_rez = Residualizer(base_design, tensorqtl_flavor=tensorqtl_flavor)
                rez = base_rez

            G_one = G_t[t: t + 1]
            H_one = H_t[t: t + 1]
            anc_stats = {k: v.detach().cpu().numpy().astype(np.float32)
                         for k, v in ancestry_allele_stats(G_one, H_one).items()}
            y_r, D_r, L_r = residualize_batch_dosage(y_t, G_one, H_one, rez, center=True, group=False)
            res = run_batch_dosage_regression(y=y_r, D=D_r, L=L_r, k_eff=rez.k_eff, device=device)
            vals, slope, slope_se, tstat, pval, dof = dosage_result_arrays(
                res, anc_stats=anc_stats, min_ancestry_ac=min_ancestry_ac
            )
            K = int(D_r.shape[2])
            var_pos = int(vsub.loc[vid, "pos"])
            rec = {
                "phenotype_id": str(pid),
                "variant_id": vid,
                "n_conditioned": len(others),
                "start_distance": int(var_pos - start_pos),
                "end_distance": int(var_pos - end_pos),
                "af": float(af_t[t].item()),
                "ma_samples": int(ma_samples_t[t].item()),
                "ma_count": float(ma_count_t[t].item()),
                "slope": float(slope[0]),
                "slope_se": float(slope_se[0]),
                "tstat": float(tstat[0]),
                "pval_nominal": float(pval[0]),
                "dof": int(dof),
            }
            if ranks:
                rec["rank"] = ranks.get((str(pid), vid), np.nan)
            for col in _dosage_columns(K):
                rec[col] = float(np.asarray(vals[col]).reshape(-1)[0])
            records.append(rec)

    if not records:
        # Same columns the non-empty path produces, so that callers can index
        # the dosage columns of a result that happens to have no rows.
        cols = ["phenotype_id", "variant_id"] + (["rank"] if ranks else [])
        cols += ["n_conditioned", "start_distance", "end_distance", "af", "ma_samples",
                 "ma_count", "slope", "slope_se", "tstat", "pval_nominal", "dof"]
        if n_anc_out:
            cols += _dosage_columns(int(n_anc_out))
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame.from_records(records)
    lead = ["phenotype_id", "variant_id"] + (["rank"] if "rank" in out.columns else [])
    out = out[lead + [c for c in out.columns if c not in lead]]
    logger.write(f"  * fitted {len(out)} signals")
    return out
