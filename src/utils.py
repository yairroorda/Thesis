import logging
import shutil
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
from rich.console import Console

LOGGER_LEVEL = logging.DEBUG

console = Console()


@contextmanager
def status_spinner(message="Processing..."):
    """Context manager to show a rich spinner during long tasks."""
    with console.status(message) as status:
        yield status


def timed(label="No label provided"):
    """Decorator that logs elapsed time for a function call."""

    def decorator(func):
        display = label or func.__name__

        @wraps(func)  # Used to preserve the original function's metadata (like name and docstring)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                log = get_logger("Timing")
                log.info(f"{display}: {elapsed:.3f}s")

        return wrapper

    return decorator


def get_logger(name="thesis", logfile_path=None, level=None):
    """Return a configured logger (idempotent). Optionally add a file handler."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(levelname)s] | %(name)s | %(message)s")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    if logfile_path:
        file_target = str(Path(logfile_path).resolve())
        has_file_handler = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == file_target for h in logger.handlers)
        if not has_file_handler:
            file_handler = logging.FileHandler(file_target)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(LOGGER_LEVEL)
    return logger


def prepare_run_folder(base_dir: str | Path, run_name: str, overwrite: bool = False) -> Path:
    """Create isolated run folder data/<run_name>, with overwrite support."""
    folder = Path(base_dir) / run_name
    if folder.exists():
        if not overwrite:
            raise FileExistsError(f"Run folder '{folder}' already exists. Use overwrite to replace it.")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def compare(
    logger: logging.Logger,
    func_old,
    func_new,
    *args,
    runs: int = 1,
    **kwargs,
) -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(runs):
        result1 = func_old(*args, **kwargs)
    end = time.perf_counter()
    old_time = end - start

    start_2 = time.perf_counter()
    for _ in range(runs):
        result2 = func_new(*args, **kwargs)
    end_2 = time.perf_counter()
    new_time = end_2 - start_2

    speedup = old_time / new_time
    logger.info(f"Old time: {old_time:.3f}s")
    logger.info(f"New time: {new_time:.3f}s")
    logger.info(f"Speedup: {speedup:.2f}x faster")

    if np.array_equal(result1, result2):
        logger.info("Both functions produced the same results.")
    else:
        logger.error("Functions produced different results!")
        logger.debug(f"Length of result 1: {len(result1)}")
        logger.debug(f"Length of result 2: {len(result2)}")
        logger.debug(f"Result 1: {result1}")
        logger.debug(f"Result 2: {result2}")


if __name__ == "__main__":

    @timed("Example function")
    def example():
        time.sleep(1.5)

    example()
