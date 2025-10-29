from __future__ import annotations
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetSink:
    """
    Minimal streaming Parquet writer.
    Use from compute modules without pulling pyarrow everywhere.
    """
    def __init__(self, out_path: str, compression: str = "snappy"):
        self.out_path = out_path
        self.compression = compression
        self._writer = None
        self._schema = None
        self._rows = 0

    def write(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.out_path, self._schema,
                                            compression=self.compression)
        self._writer.write_table(table)
        self._rows += len(df)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

