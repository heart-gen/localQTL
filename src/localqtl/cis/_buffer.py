import numpy as np
import pandas as pd

__all__ = [
    "allocate_result_buffers",
    "ensure_capacity", "write_row",
    "buffers_to_dataframe"
]

def allocate_result_buffers(
        columns: list[str], dtype_map: dict[str, type | np.dtype],
        target_rows: int) -> dict[str, np.ndarray]:
    target = max(int(target_rows), 1)
    buffers: dict[str, np.ndarray] = {}
    for name in columns:
        dtype = dtype_map.get(name, np.float32)
        buffers[name] = np.empty(target, dtype=dtype)
    return buffers


def ensure_capacity(buffers: dict[str, np.ndarray], cursor: int,
                    additional: int) -> dict[str, np.ndarray]:
    required = cursor + int(additional)
    current = next(iter(buffers.values())).shape[0] if buffers else 0
    if required <= current:
        return buffers
    new_size = max(required, current * 2 if current else 1)
    for key, arr in buffers.items():
        new_arr = np.empty(new_size, dtype=arr.dtype)
        new_arr[:cursor] = arr[:cursor]
        buffers[key] = new_arr
    return buffers


def write_row(buffers: dict[str, np.ndarray], idx: int, row: dict[str, object]) -> None:
    for key, arr in buffers.items():
        value = row.get(key)
        if arr.dtype == object:
            arr[idx] = None if value is None else value
        elif np.issubdtype(arr.dtype, np.integer):
            arr[idx] = int(value) if value is not None else 0
        else:
            arr[idx] = np.nan if value is None else float(value)


def buffers_to_dataframe(columns: list[str], buffers: dict[str, np.ndarray],
                         n_rows: int) -> pd.DataFrame:
    n = int(n_rows)
    data = {col: buffers[col][:n] for col in columns if col in buffers}
    return pd.DataFrame(data, columns=columns)
