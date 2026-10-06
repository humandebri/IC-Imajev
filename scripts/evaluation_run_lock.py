"""Serialize evaluation processes sharing a run directory, including report reads."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the inode in place. The OS releases this lock even if a worker exits.
    with (directory / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
