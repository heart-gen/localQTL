## SimpleLogger adapted from tensorqtl:
## https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
from __future__ import annotations
import logging
import sys, time, hashlib
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

__all__ = [
    "SimpleLogger",
    "NullLogger",
    "gpu_available",
    "pick_device",
    "select_array_module",
    "select_dataframe_backend",
    "select_dask_array_backend",
    "subseed",
]

_GPU_ARRAY_AVAILABLE: Optional[bool] = None
_GPU_FRAME_AVAILABLE: Optional[bool] = None
_GPU_DASK_AVAILABLE: Optional[bool] = None

class SimpleLogger:
    def __init__(self, logfile: Optional[str] = None, verbose: bool = True,
                 timestamps: bool = False, timefmt: str = "%Y-%m-%d %H:%M:%S"):
        self.console = sys.stdout
        self.verbose = verbose
        self.log = open(logfile, "w") if logfile else None
        self.timestamps = timestamps
        self.timefmt = timefmt

    def _stamp(self, msg: str) -> str:
        if self.timestamps:
            return f"[{datetime.now().strftime(self.timefmt)}] {msg}"
        return msg

    def write(self, message: str):
        line = self._stamp(message)
        if self.verbose:
            self.console.write(line + "\n")
        if self.log is not None:
            self.log.write(line + "\n")
            self.log.flush()

    @contextmanager
    def time_block(self, label: str, sync=None, sec=True):
        if sync: sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync: sync()
            dt = time.perf_counter() - t0
            if sec:
                self.write(f"{label} done in {dt:.2f}s")
            else:
                self.write(f"{label} done in {dt / 60: .2f} min")

    def close(self):
        if self.log:
            try: self.log.close()
            finally: self.log = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class NullLogger(SimpleLogger):
    def __init__(self): super().__init__(logfile=None, verbose=False)
    def write(self, message: str): pass


def gpu_available():
    cp = _try_get_cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _try_get_cupy(force_gpu: bool = False):
    global _GPU_ARRAY_AVAILABLE
    if _GPU_ARRAY_AVAILABLE is False and not force_gpu:
        return None
    try:
        import cupy as cp
        # Trigger CUDA runtime check early to catch driver issues.
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        _GPU_ARRAY_AVAILABLE = False
        if force_gpu:
            raise RuntimeError(
                "GPU array backend requested but CuPy is unavailable or failed to initialize."
            ) from e
        logging.warning(
            "CuPy unavailable or failed to initialize; falling back to NumPy CPU backend."
        )
        return None

    _GPU_ARRAY_AVAILABLE = True
    return cp


def _try_get_cudf(force_gpu: bool = False):
    global _GPU_FRAME_AVAILABLE
    if _GPU_FRAME_AVAILABLE is False and not force_gpu:
        return None
    try:
        import cudf
    except Exception as e:
        _GPU_FRAME_AVAILABLE = False
        if force_gpu:
            raise RuntimeError(
                "GPU dataframe backend requested but cuDF is unavailable or failed to initialize."
            ) from e
        logging.warning(
            "cuDF unavailable or failed to initialize; falling back to pandas CPU backend."
        )
        return None

    _GPU_FRAME_AVAILABLE = True
    return cudf


def _try_get_dask_cudf(force_gpu: bool = False):
    global _GPU_DASK_AVAILABLE
    if _GPU_DASK_AVAILABLE is False and not force_gpu:
        return None
    try:
        import dask_cudf
    except Exception as e:
        _GPU_DASK_AVAILABLE = False
        if force_gpu:
            raise RuntimeError(
                "GPU dask dataframe backend requested but dask-cudf is unavailable or failed to initialize."
            ) from e
        logging.warning(
            "dask-cudf unavailable or failed to initialize; falling back to dask CPU backend."
        )
        return None

    _GPU_DASK_AVAILABLE = True
    return dask_cudf


def select_array_module(force_gpu: bool = False):
    cp = _try_get_cupy(force_gpu=force_gpu)
    if cp is not None:
        return cp
    import numpy as np
    return np


def select_dataframe_backend(force_gpu: bool = False):
    cudf = _try_get_cudf(force_gpu=force_gpu)
    if cudf is not None:
        from cudf import DataFrame as cuDF
        return cudf, cuDF
    import pandas as pd
    return pd, pd.DataFrame


def select_dask_array_backend(force_gpu: bool = False):
    dask_cudf = _try_get_dask_cudf(force_gpu=force_gpu)
    if dask_cudf is not None:
        return dask_cudf
    import dask.array as da
    return da


def pick_device(prefer: str = "auto") -> str:
    if prefer in {"cpu", "cuda"}:
        return prefer if (prefer != "cuda" or gpu_available()) else "cpu"
    return "cuda" if gpu_available() else "cpu"


def subseed(base: int, key: str | int) -> int:
    """Deterministic 64-bit sub-seed from base seed and a stable key (pid/group)."""
    h = int(hashlib.blake2b(str(key).encode(), digest_size=8).hexdigest(), 16)
    return (h ^ int(base)) & ((1 << 63) - 1)


