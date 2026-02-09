import time
from functools import wraps


def timed(label="No label provided"):
    """Decorator that prints elapsed time for a function call."""

    def decorator(func):
        display = label or func.__name__

        @wraps(func) # Used to preserve the original function's metadata (like name and docstring)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                print(f"{display}: {elapsed:.1f}s")

        return wrapper

    return decorator

if __name__ == "__main__":
    @timed("Example function")
    def example():
        time.sleep(1.5)

    example()