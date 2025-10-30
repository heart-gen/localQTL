from __future__ import annotations
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "ParquetSink",
]

class ParquetSink:
    """
    Minimal streaming Parquet writer with stable schema enforcement.
    """
    def __init__(self, out_path: str, compression: str = "snappy",
                 overwrite: bool = True, row_group_size: int | None = None,
                 schema: pa.Schema | None = None, ensure_parent: bool = True):
        self.out_path = out_path
        self.compression = compression
        self.overwrite = overwrite
        self.row_group_size = row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = schema
        self._rows: int = 0
        self._closed: bool = False

        if ensure_parent:
            parent = os.path.dirname(out_path) or "."
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(out_path) and overwrite:
            os.remove(out_path)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def closed(self) -> bool:
        return self._closed

    def _align_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reorder/add missing columns to match schema (fill with NA)
        cols = [f.name for f in self._schema]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = pd.NA
        # Drop extras (keep only schema columns) and reorder
        return df[cols]

    def write(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return

        if self._schema is None: # First batch defines schema
            table = pa.Table.from_pandas(df, preserve_index=False)
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.out_path, self._schema, compression=self.compression
            )
            if self.row_group_size:
                self._writer.write_table(table, row_group_size=self.row_group_size)
            else:
                self._writer.write_table(table)
            self._rows += len(df)
            return

        df2 = self._align_to_schema(df)
        table = pa.Table.from_pandas(df2, schema=self._schema, preserve_index=False)
        if not table.schema.equals(self._schema, check_metadata=False):
            table = table.cast(self._schema)

        if self.row_group_size:
            self._writer.write_table(table, row_group_size=self.row_group_size)
        else:
            self._writer.write_table(table)
        self._rows += len(df)

    def close(self) -> None:
        if self._writer is not None and not self._closed:
            self._writer.close()
            self._writer = None
            self._closed = True

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
