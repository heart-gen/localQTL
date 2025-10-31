from __future__ import annotations
import os
import queue, threading
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["ParquetSink", "AsyncParquetSink", "RowGroupBuffer"]

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
        if isinstance(use_dictionary, Iterable) and not isinstance(use_dictionary, (bool, str)):
            use_dictionary = list(use_dictionary)
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

        if self._closed:
            raise RuntimeError("Cannot write to a closed ParquetSink")

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


class AsyncParquetSink:
    """
    Background-thread writer that wraps ParquetSink.
    Benefits:
      - Overlaps Arrow compression + filesystem writes with your GPU/CPU compute
      - Bounded queue applies backpressure (keeps memory in check)
      - Error in writer thread is re-raised on close()
    """
    def __init__(
            self,
            out_path: str,
            *,
            schema,
            compression: str = "snappy",
            row_group_size: int = 1_000_000,
            use_dictionary: bool | Iterable[str] | None = None,
            write_statistics: bool = False,
            column_order: Optional[List[str]] = None,
            max_queue_items: int = 8,
            overwrite: bool = True,
            ensure_parent: bool = True,
    ):
        self._sink = ParquetSink(
            out_path=out_path,
            compression=compression,
            overwrite=overwrite,
            row_group_size=row_group_size,
            schema=schema,
            ensure_parent=ensure_parent,
            use_dictionary=use_dictionary,
            write_statistics=write_statistics,
        )
        self._column_order = column_order
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_items)
        self._stop = object()
        self._exc: BaseException | None = None
        self._rows = 0
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    @property
    def rows(self) -> int:
        return self._rows

    def _infer_rows(self, item: Any) -> int:
        if isinstance(item, dict):
            first = next(iter(item.values()))
            return len(first)
        try:
            # pa.Table
            return item.num_rows  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            # pandas.DataFrame
            return len(item)  # type: ignore[arg-type]
        except Exception:
            return 0

    def write(self, data: Any) -> None:
        # If the writer thread already failed, surface it now.
        if self._exc is not None:
            raise RuntimeError("AsyncParquetSink writer failed") from self._exc
        self._q.put(data, block=True)

    def _worker(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is self._stop:
                    break
                self._sink.write(item, column_order=self._column_order)
                self._rows += self._infer_rows(item)
        except BaseException as e:
            self._exc = e
        finally:
            # Let close() handle finalization and surfacing errors
            pass

    def close(self) -> None:
        # Signal stop and join the thread
        try:
            self._q.put(self._stop, block=True)
        except Exception:
            pass
        self._thr.join()
        # Close underlying sink first
        try:
            self._sink.close()
        finally:
            # If writer thread failed, re-raise here so callers see it
            if self._exc is not None:
                raise RuntimeError("AsyncParquetSink writer failed") from self._exc

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()


# iosinks.py  (place below AsyncParquetSink)

import numpy as np
from typing import Dict, List, Optional

class RowGroupBuffer:
    """
    Accumulates batches (dict[str, np.ndarray]) until a target row count is reached,
    then hands back a single concatenated dict ready for ParquetSink.write().
    Not thread-safe; use from the producer thread.
    """
    def __init__(self, column_order: List[str], target_rows: int = 1_000_000):
        self._cols = list(column_order)
        self._target = int(target_rows)
        self._buf: Dict[str, List[np.ndarray]] = {c: [] for c in self._cols}
        self._n = 0

    @property
    def size(self) -> int:
        return self._n

    def add(self, batch: Dict[str, np.ndarray]) -> None:
        if not batch:
            return
        n = len(next(iter(batch.values())))
        # assume caller provides all columns in column_order
        for k in self._cols:
            self._buf[k].append(batch[k])
        self._n += n

    def ready(self) -> bool:
        return self._n >= self._target

    def take(self) -> Optional[Dict[str, np.ndarray]]:
        if self._n == 0:
            return None
        out = {
            k: v[0] if len(v) == 1 else np.concatenate(v, axis=0)
            for k, v in self._buf.items()
        }
        # reset
        self._buf = {c: [] for c in self._cols}
        self._n = 0
        return out
