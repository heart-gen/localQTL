"""
This script was adapted from tensorQTL `core.py`:
https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
"""
import pandas as pd

def read_phenotype_bed(path):
    """
    Load phenotype BED file into phenotype and position DataFrames.

    Parameters
    ----------
    path : str
        Path to BED-like file (.bed, .bed.gz, .bed.parquet, .parquet).

    Returns
    -------
    phenotype_df : pd.DataFrame
        Phenotype matrix (rows = phenotypes, columns = samples).
    pos_df : pd.DataFrame
        Position metadata with columns:
        - 'chr'
        - 'pos' (if start == end)
        - or ['chr','start','end'] otherwise
    """
    # Load file
    if path.lower().endswith(('.bed.gz', '.bed')):
        df = pd.read_csv(
            path, sep="\t", dtype={"#chr": str, "#Chr": str}
        )
    elif path.lower().endswith(('.bed.parquet', '.parquet')):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Normalize columns
    cols = [c.lower().replace("#chr", "chr").replace("#", "") for c in df.columns]
    df.columns = cols

    # Ensure phenotype ID is index
    if df.columns[3] != "id":
        df.rename(columns={df.columns[3]: "id"}, inplace=True)
    df.set_index("id", inplace=True)

    # Adjust coordinates
    if "start" in df.columns:
        df["start"] = df["start"].astype(int) + 1  # 1-based

    # Build position table
    if {"chr", "start", "end"}.issubset(df.columns):
        pos_df = df[["chr", "start", "end"]].copy()
        df = df.drop(["chr", "start", "end"], axis=1)
    else:
        raise ValueError("BED file must contain 'chr', 'start', 'end' columns")

    #  Sort check
    sorted_pos = pos_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)
    if not pos_df.reset_index(drop=True).equals(sorted_pos):
        raise ValueError("Positions in BED file must be sorted by chr/start/end")

    # Collapse start==end to single pos
    if (pos_df["start"] == pos_df["end"]).all():
        pos_df = pos_df[["chr", "end"]].rename(columns={"end": "pos"})

    return df, pos_df
