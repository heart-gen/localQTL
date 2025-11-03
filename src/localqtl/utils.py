## SimpleLogger adapted from tensorqtl:
## https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
from __future__ import annotations
import sys, time, hashlib
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

__all__ = [
    "SimpleLogger",
    "NullLogger",
    "gpu_available",
    "pick_device",
    "subseed",
]

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
    try:
        import cupy as cp
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            return False
        return ndev > 0
    except ModuleNotFoundError:
        return False


def pick_device(prefer: str = "auto") -> str:
    if prefer in {"cpu", "cuda"}:
        return prefer if (prefer != "cuda" or gpu_available()) else "cpu"
    return "cuda" if gpu_available() else "cpu"


def subseed(base: int, key: str | int) -> int:
    """Deterministic 64-bit sub-seed from base seed and a stable key (pid/group)."""
    h = int(hashlib.blake2b(str(key).encode(), digest_size=8).hexdigest(), 16)
    return (h ^ int(base)) & ((1 << 63) - 1)


