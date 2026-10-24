"""
Standardised external allele-frequency tables.

A :class:`FrequencyTable` holds, per variant, allele frequency (``af_<group>``),
allele count (``ac_<group>``) and allele number (``an_<group>``) for one or more
reference groups (gnomAD genetic-ancestry groups, gnomAD local-ancestry tracts,
ALFA populations, ...).  Variants are keyed by ``(chrom, pos, ref, alt)`` on
GRCh38 and ``af`` always refers to ``alt``.  :meth:`FrequencyTable.lookup`
joins the table onto a localQTL ``variant_df`` allele-aware (exact or swapped
ref/alt) and flags strand-ambiguous and unmatched variants.

Mapping the local-ancestry panel labels (e.g. ``"AFR"``) onto table groups is
always explicit through ``ancestry_map={panel_label: group_key}``; see
:func:`resolve_ancestry_map`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GRCH38_REFSEQ_TO_CHROM",
    "CHROM_TO_GRCH38_REFSEQ",
    "STRAND_AMBIGUOUS",
    "KEY_COLUMNS",
    "normalize_chrom",
    "is_strand_ambiguous",
    "ensure_variant_keys",
    "FrequencyTable",
    "read_frequency_table",
    "resolve_ancestry_map",
]

logger = logging.getLogger(__name__)

# GRCh38 primary-assembly RefSeq accessions (NCBI assembly report GCF_000001405.40).
GRCH38_REFSEQ_TO_CHROM: dict[str, str] = {
    "NC_000001.11": "1", "NC_000002.12": "2", "NC_000003.12": "3", "NC_000004.12": "4",
    "NC_000005.10": "5", "NC_000006.12": "6", "NC_000007.14": "7", "NC_000008.11": "8",
    "NC_000009.12": "9", "NC_000010.11": "10", "NC_000011.10": "11", "NC_000012.12": "12",
    "NC_000013.11": "13", "NC_000014.9": "14", "NC_000015.10": "15", "NC_000016.10": "16",
    "NC_000017.11": "17", "NC_000018.10": "18", "NC_000019.10": "19", "NC_000020.11": "20",
    "NC_000021.9": "21", "NC_000022.11": "22", "NC_000023.11": "X", "NC_000024.10": "Y",
    "NC_012920.1": "M",
}
CHROM_TO_GRCH38_REFSEQ: dict[str, str] = {v: k for k, v in GRCH38_REFSEQ_TO_CHROM.items()}

STRAND_AMBIGUOUS = {frozenset(("A", "T")), frozenset(("C", "G"))}
KEY_COLUMNS = ("chrom", "pos", "ref", "alt")
_METADATA_KEY = b"localqtl.freq"


def normalize_chrom(value, style: str = "chr") -> Optional[str]:
    """
    Canonical chromosome label.

    Accepts ``"chr7"``, ``"7"``, ``"chrM"``, ``"MT"`` and GRCh38 RefSeq
    accessions (``"NC_000007.14"``).  Returns ``None`` for accessions that are
    not GRCh38 primary chromosomes (e.g. GRCh37 ``NC_000007.13``), so callers
    can drop them.  ``style="chr"`` gives ``chr7``; ``style="plain"`` gives ``7``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s.startswith("NC_"):
        core = GRCH38_REFSEQ_TO_CHROM.get(s)
        if core is None:
            return None
    else:
        core = s[3:] if s.lower().startswith("chr") else s
        core = core.upper()
        if core == "MT":
            core = "M"
    return f"chr{core}" if style == "chr" else core


def is_strand_ambiguous(ref, alt) -> np.ndarray:
    """Element-wise A/T or C/G flag for ``ref``/``alt`` arrays."""
    ref = pd.Series(ref).astype(str).str.upper().to_numpy()
    alt = pd.Series(alt).astype(str).str.upper().to_numpy()
    return np.array([frozenset((r, a)) in STRAND_AMBIGUOUS for r, a in zip(ref, alt)], dtype=bool)


def ensure_variant_keys(variant_df: pd.DataFrame, *, require_alleles: bool = True) -> pd.DataFrame:
    """
    Return ``chrom, pos, ref, alt`` (canonical chrom, int pos, upper-case
    alleles) aligned to ``variant_df.index``.  When ``ref``/``alt`` are missing
    they are parsed from IDs shaped like ``chr1_123_A_G_b38`` via
    :func:`localqtl.genotypeio.alleles_from_variant_ids`; with
    ``require_alleles=False`` unparsable IDs leave NaN alleles.
    """
    df = variant_df
    if not {"chrom", "pos"}.issubset(df.columns):
        raise ValueError("variant_df needs 'chrom' and 'pos' columns.")
    if not {"ref", "alt"}.issubset(df.columns):
        from ..genotypeio import alleles_from_variant_ids
        df = alleles_from_variant_ids(df)
        if require_alleles and df[["ref", "alt"]].isna().any().any():
            n_bad = int(df[["ref", "alt"]].isna().any(axis=1).sum())
            raise ValueError(
                f"{n_bad} variants have no ref/alt alleles; add 'ref'/'alt' columns to variant_df "
                "(PgenReader / PlinkReader.variant_df provide them) or use position_only=True.")
    keys = pd.DataFrame(index=df.index)
    keys["chrom"] = [normalize_chrom(c) for c in df["chrom"].to_numpy()]
    keys["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64").astype("int64")
    keys["ref"] = df["ref"].astype("string").str.upper()
    keys["alt"] = df["alt"].astype("string").str.upper()
    return keys


def _group_columns(columns: Iterable[str]) -> list[str]:
    groups = []
    for c in columns:
        if c.startswith("af_"):
            groups.append(c[3:])
    return groups


@dataclass
class FrequencyTable:
    """
    External allele frequencies keyed by ``(chrom, pos, ref, alt)``.

    Parameters
    ----------
    df : DataFrame
        Columns ``chrom, pos, ref, alt`` (optional ``rsid``) plus ``af_<group>``
        and optionally ``ac_<group>``, ``an_<group>`` for each group.
    source : str
        Free-text provenance (``"gnomad_v4.1_genomes"``, ``"alfa_r5"``, ...).
    build : str
        Genome build; only ``"GRCh38"`` is accepted.
    metadata : dict
        Anything else worth keeping (header fields, tract tags, URLs).
    """
    df: pd.DataFrame
    source: str = "unknown"
    build: str = "GRCh38"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.build != "GRCh38":
            raise ValueError(f"Only GRCh38 frequency tables are supported (got {self.build!r}).")
        missing = [c for c in KEY_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Frequency table is missing key columns: {missing}")
        df = self.df.copy()
        df["chrom"] = [normalize_chrom(c) for c in df["chrom"].to_numpy()]
        dropped = df["chrom"].isna()
        if dropped.any():
            logger.warning("Dropping %d rows with non-GRCh38 chromosome labels", int(dropped.sum()))
            df = df.loc[~dropped]
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("int64")
        df["ref"] = df["ref"].astype("string").str.upper()
        df["alt"] = df["alt"].astype("string").str.upper()
        if not self.groups_from(df):
            raise ValueError("Frequency table has no 'af_<group>' columns.")
        dup = df.duplicated(list(KEY_COLUMNS), keep="first")
        if dup.any():
            logger.warning("Dropping %d duplicated (chrom,pos,ref,alt) rows", int(dup.sum()))
            df = df.loc[~dup]
        self.df = df.reset_index(drop=True)

    # -- introspection -----------------------------------------------------
    @staticmethod
    def groups_from(df: pd.DataFrame) -> list[str]:
        return _group_columns(df.columns)

    @property
    def groups(self) -> list[str]:
        """Group keys present (suffixes of the ``af_`` columns)."""
        return self.groups_from(self.df)

    def __len__(self) -> int:
        return int(len(self.df))

    def validate_groups(self, groups: Iterable[str]) -> list[str]:
        groups = list(groups)
        unknown = [g for g in groups if g not in self.groups]
        if unknown:
            raise ValueError(f"Unknown frequency group(s) {unknown}; available: {self.groups}")
        return groups

    def select_groups(self, groups: Iterable[str]) -> "FrequencyTable":
        groups = self.validate_groups(groups)
        keep = [c for c in self.df.columns if not any(c.startswith(p) for p in ("af_", "ac_", "an_"))]
        for g in groups:
            keep.extend(c for c in (f"af_{g}", f"ac_{g}", f"an_{g}") if c in self.df.columns)
        return FrequencyTable(self.df[keep].copy(), source=self.source, build=self.build,
                              metadata=dict(self.metadata))

    def rename_groups(self, mapping: Mapping[str, str]) -> "FrequencyTable":
        self.validate_groups(mapping.keys())
        ren = {}
        for old, new in mapping.items():
            for p in ("af_", "ac_", "an_"):
                if f"{p}{old}" in self.df.columns:
                    ren[f"{p}{old}"] = f"{p}{new}"
        return FrequencyTable(self.df.rename(columns=ren), source=self.source, build=self.build,
                              metadata=dict(self.metadata))

    # -- I/O ---------------------------------------------------------------
    def write(self, path: str) -> str:
        """Write Parquet (default) or TSV (``.tsv``/``.txt``/``.gz``) with metadata."""
        path = str(path)
        if path.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")):
            with pd.option_context("mode.copy_on_write", True):
                self.df.to_csv(path, sep="\t", index=False)
            side = path + ".json"
            with open(side, "w") as fh:
                json.dump({"source": self.source, "build": self.build, "metadata": self.metadata}, fh)
            return path
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(self.df, preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta[_METADATA_KEY] = json.dumps(
            {"source": self.source, "build": self.build, "metadata": self.metadata}).encode()
        pq.write_table(table.replace_schema_metadata(meta), path)
        return path

    # -- joins -------------------------------------------------------------
    def lookup(self, variant_df: pd.DataFrame, *, groups: Optional[Sequence[str]] = None,
               position_only: bool = False) -> pd.DataFrame:
        """
        Allele-aware join onto ``variant_df`` (index = variant IDs).

        Matching order: exact ``(chrom,pos,ref,alt)``; then swapped alleles
        (``af`` becomes ``1 - af`` and ``ac`` becomes ``an - ac`` so every column
        refers to ``variant_df.alt``); with ``position_only=True`` the remaining
        rows match on ``(chrom,pos)`` when the table has exactly one row there
        (no orientation adjustment; use only when alleles are unavailable).

        Returns a DataFrame aligned to ``variant_df.index`` with the group
        columns plus ``match_type`` (``exact``/``flipped``/``position_only``/
        ``none``), ``allele_flipped``, ``strand_ambiguous``,
        ``missing_in_reference`` and ``ref_alleles`` (table ``ref/alt``).
        """
        groups = self.validate_groups(groups) if groups is not None else self.groups
        value_cols = [c for g in groups for c in (f"af_{g}", f"ac_{g}", f"an_{g}") if c in self.df.columns]
        keys = ensure_variant_keys(variant_df, require_alleles=not position_only)
        keys = keys.reset_index().rename(columns={keys.index.name or "index": "_vid"})
        if "_vid" not in keys.columns:          # index had a name other than 'index'
            keys = keys.rename(columns={keys.columns[0]: "_vid"})
        tab = self.df[list(KEY_COLUMNS) + value_cols].copy()
        tab["ref_alleles"] = tab["ref"].astype(str) + "/" + tab["alt"].astype(str)

        out = pd.DataFrame({"_vid": keys["_vid"]})
        for c in value_cols:
            out[c] = np.nan
        out["match_type"] = "none"
        out["ref_alleles"] = pd.array([None] * len(out), dtype="string")

        has_alleles = keys["ref"].notna() & keys["alt"].notna()

        # 1) exact
        m = keys.loc[has_alleles].merge(tab, on=list(KEY_COLUMNS), how="inner")
        self._fill(out, m, value_cols, "exact")
        # 2) flipped
        todo = has_alleles & (out["match_type"] == "none")
        if todo.any():
            swapped = tab.rename(columns={"ref": "alt", "alt": "ref"})
            m = keys.loc[todo].merge(swapped, on=list(KEY_COLUMNS), how="inner")
            for g in groups:
                if f"af_{g}" in m.columns:
                    m[f"af_{g}"] = 1.0 - m[f"af_{g}"]
                if f"ac_{g}" in m.columns and f"an_{g}" in m.columns:
                    m[f"ac_{g}"] = m[f"an_{g}"] - m[f"ac_{g}"]
            self._fill(out, m, value_cols, "flipped")
        # 3) position only
        if position_only:
            todo = out["match_type"] == "none"
            if todo.any():
                uniq = tab.groupby(["chrom", "pos"]).filter(lambda x: len(x) == 1)
                m = keys.loc[todo, ["_vid", "chrom", "pos"]].merge(uniq, on=["chrom", "pos"], how="inner")
                self._fill(out, m, value_cols, "position_only")

        out["allele_flipped"] = out["match_type"] == "flipped"
        out["missing_in_reference"] = out["match_type"] == "none"
        amb = np.zeros(len(out), dtype=bool)
        if has_alleles.any():
            amb[has_alleles.to_numpy()] = is_strand_ambiguous(keys.loc[has_alleles, "ref"],
                                                              keys.loc[has_alleles, "alt"])
        out["strand_ambiguous"] = amb
        out = out.set_index("_vid")
        out.index.name = variant_df.index.name
        return out.loc[variant_df.index]

    @staticmethod
    def _fill(out: pd.DataFrame, matched: pd.DataFrame, value_cols: list[str], label: str) -> None:
        if matched.empty:
            return
        matched = matched.drop_duplicates("_vid")
        pos = out.index[out["_vid"].isin(matched["_vid"])]
        m = matched.set_index("_vid").loc[out.loc[pos, "_vid"]]
        for c in value_cols:
            out.loc[pos, c] = m[c].to_numpy(dtype=float)
        out.loc[pos, "match_type"] = label
        out.loc[pos, "ref_alleles"] = m["ref_alleles"].to_numpy()


def read_frequency_table(path: str, *, source: Optional[str] = None,
                         build: Optional[str] = None) -> FrequencyTable:
    """Read a table written by :meth:`FrequencyTable.write` (Parquet or TSV)."""
    path = str(path)
    meta = {"source": source or "unknown", "build": build or "GRCh38", "metadata": {}}
    if path.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")):
        df = pd.read_csv(path, sep="\t", dtype={"chrom": str, "ref": str, "alt": str})
        side = path + ".json"
        if os.path.exists(side):
            with open(side) as fh:
                saved = json.load(fh)
            meta.update({k: v for k, v in saved.items() if v})
    else:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        if raw:
            meta.update({k: v for k, v in json.loads(raw.decode()).items() if v})
        df = table.to_pandas()
    if source:
        meta["source"] = source
    if build:
        meta["build"] = build
    return FrequencyTable(df, source=meta["source"], build=meta["build"], metadata=meta["metadata"])


def resolve_ancestry_map(ancestry_map: Mapping[str, str], panel_labels: Sequence[str],
                         table: FrequencyTable) -> dict[str, str]:
    """
    Validate ``{panel_label: table_group}``.  Every key must be a panel
    ancestry label and every value a group in ``table``; there are no defaults.
    """
    if not isinstance(ancestry_map, Mapping) or not ancestry_map:
        raise ValueError(
            "ancestry_map is required: a dict {panel_label: table_group}, e.g. "
            f"{{{', '.join(repr(l) + ': <group>' for l in panel_labels)}}}; "
            f"table groups: {table.groups}")
    labels = [str(l) for l in panel_labels]
    bad_keys = [k for k in ancestry_map if str(k) not in labels]
    if bad_keys:
        raise ValueError(f"ancestry_map keys {bad_keys} are not panel ancestries {labels}")
    table.validate_groups(ancestry_map.values())
    return {str(k): str(v) for k, v in ancestry_map.items()}
