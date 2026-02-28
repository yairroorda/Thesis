import logging
import time
from functools import wraps

import numpy as np

LOGGER_LEVEL = logging.DEBUG


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


def get_logger(name="thesis"):
    """Return a configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(LOGGER_LEVEL)
    return logger


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
