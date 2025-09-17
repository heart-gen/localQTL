import sys
from time import strftime

class SimpleLogger(object):
    """
    Simple logger that writes timestamped messages to console and optionally to a file.
    Supports context manager usage to ensure logfile is always closed.

    Example
    -------
    with SimpleLogger("analysis.log") as logger:
        logger.write("Starting cis-mapping")
    """

    def __init__(self, logfile: str = None, verbose: bool = True):
        self.console = sys.stdout
        self.verbose = verbose
        self.log = open(logfile, "w") if logfile else None

    def _timestamp(self) -> str:
        """Return current timestamp as string."""
        return strftime("%Y-%m-%d %H:%M:%S")

    def write(self, message: str):
        """Write a timestamped message to console and logfile (if provided)."""
        msg = f"[{self._timestamp()}] {message}"
        if self.verbose:
            self.console.write(msg + "\n")
        if self.log is not None:
            self.log.write(msg + "\n")
            self.log.flush()

    def close(self):
        """Close the logfile if open."""
        if self.log is not None:
            self.log.close()
            self.log = None

    # Context manager methods
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def gpu_available():
    import cupy as cp
    try:
        ndev = cp.cuda.runtime.getDeviceCount()
        return ndev > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False
