import logging
import time
from functools import wraps

LOGGER_LEVEL = logging.DEBUG

def timed(label="No label provided"):
    """Decorator that logs elapsed time for a function call."""

    def decorator(func):
        display = label or func.__name__

        @wraps(func) # Used to preserve the original function's metadata (like name and docstring)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                log = get_logger("Timing")
                log.info(f"{display}: {elapsed:.1f}s")

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

if __name__ == "__main__":
    @timed("Example function")
    def example():
        time.sleep(1.5)

    example()