## SimpleLogger adapted from tensorqtl:
## https://github.com/broadinstitute/tensorqtl/blob/master/tensorqtl/core.py
from __future__ import annotations
import sys, time
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

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
    def time_block(self, label: str, sync=None):
        if sync: sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync: sync()
            dt = time.perf_counter() - t0
            self.write(f"{label} done in {dt:.2f}s")

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
    import cupy as cp
    try:
        ndev = cp.cuda.runtime.getDeviceCount()
        return ndev > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def pick_device(prefer: str = "auto") -> str:
    if prefer in {"cpu", "cuda"}:
        return prefer if (prefer != "cuda" or gpu_available()) else "cpu"
    return "cuda" if gpu_available() else "cpu"
