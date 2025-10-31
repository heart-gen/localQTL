from __future__ import annotations
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["ParquetSink"]

class ParquetSink:
    """
    Streaming Parquet writer (Arrow-first) with stable schema, large row-groups,
    and optional per-column dictionary encoding.

    Fastest path: pass a dict[str, np.ndarray | pa.Array] or a pa.Table.
    """
    def __init__(
        self,
        out_path: str,
        compression: str = "snappy",
        overwrite: bool = True,
        row_group_size: int | None = 1_000_000,
        schema: pa.Schema | None = None,
        ensure_parent: bool = True,
        use_dictionary: bool | Iterable[str] | None = None,
        write_statistics: bool = False,
    ):
        self.out_path = out_path
        self.compression = compression
        self.overwrite = overwrite
        self.row_group_size = row_group_size
        self.use_dictionary = use_dictionary
        self.write_statistics = write_statistics

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

    @staticmethod
    def _table_from_any(data: Any, column_order: list[str] | None = None) -> pa.Table:
        """Convert pd.DataFrame | dict | pa.Table into pa.Table without guessing."""
        if isinstance(data, pa.Table):
            return data

        if isinstance(data, dict):
            names = column_order or list(data.keys())
            arrays = []
            for k in names:
                v = data[k]
                if isinstance(v, pa.Array):
                    arrays.append(v)
                else:
                    arrays.append(pa.array(v))
            return pa.Table.from_arrays(arrays, names=names)

        if isinstance(data, pd.DataFrame):
            # When DataFrame is unavoidable; still Arrow-ize once.
            return pa.Table.from_pandas(data, preserve_index=False)

        raise TypeError(f"Unsupported input type: {type(data)}")

    def _ensure_writer(self, table: pa.Table) -> None:
        if self._writer is not None:
            return
        if self._schema is None:
            self._schema = table.schema
        self._writer = pq.ParquetWriter(
            self.out_path,
            self._schema,
            compression=self.compression,
            use_dictionary=self.use_dictionary,
            write_statistics=self.write_statistics,
        )

    def _align_and_cast(self, table: pa.Table) -> pa.Table:
        """
        Ensure columns exist, are ordered like self._schema, and cast to target types.
        Missing columns are filled with nulls of the right type.
        """
        assert self._schema is not None
        nrows = table.num_rows
        cols: list[pa.Array] = []
        for field in self._schema:
            name, ty = field.name, field.type
            if name in table.column_names:
                col = table.column(name)
                if col.type != ty:
                    col = col.cast(ty)
                cols.append(col)
            else:
                cols.append(pa.nulls(nrows, type=ty))
        return pa.Table.from_arrays(cols, schema=self._schema)

    def write(self, data: Any, column_order: list[str] | None = None) -> None:
        """
        Write a batch. `data` can be pa.Table, dict[str, array], or pd.DataFrame.
        If you pass dict/Arrow with numeric dtypes, this stays on the fast path.
        """
        if data is None:
            return

        table = self._table_from_any(data, column_order=column_order)
        if table.num_rows == 0:
            return

        if self._writer is None:
            # First batch defines (or validates) schema
            if self._schema is None:
                self._schema = table.schema
            table = self._align_and_cast(table)
            self._ensure_writer(table)
        else:
            table = self._align_and_cast(table)

        if self.row_group_size:
            # Split into large row groups (fewer = faster)
            n = table.num_rows
            rgsz = int(self.row_group_size)
            for off in range(0, n, rgsz):
                self._writer.write_table(table.slice(off, rgsz))
                self._rows += min(rgsz, n - off)
        else:
            self._writer.write_table(table)
            self._rows += table.num_rows

    def close(self) -> None:
        if self._writer is not None and not self._closed:
            self._writer.close()
            self._writer = None
            self._closed = True

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
