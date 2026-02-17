import logging
import time
import numpy as np
from functools import wraps

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


def compare_speed(
    logger: logging.Logger,
    func_old,
    func_new,
    *args,
    runs: int = 1000,
    **kwargs,
) -> dict[str, float]:
    start = time.perf_counter()
    for _ in range(runs):
        func_old(*args, **kwargs)
    end = time.perf_counter()
    tot_old = end - start

    start_2 = time.perf_counter()
    for _ in range(runs):
        func_new(*args, **kwargs)
    end_2 = time.perf_counter()
    tot_new = end_2 - start_2

    logger.info(
        f"Total runtime over {runs} runs - old: {tot_old:.6f} s, new: {tot_new:.6f} s \n new/old ratio was {tot_new / tot_old}",
    )
    return {"old": tot_old, "new": tot_new}


def compare_outcomes(logger, func1, func2):
    result1 = func1()
    result2 = func2()
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
