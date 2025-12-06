import time
from functools import wraps


def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f'function {func.__name__!r} executed in {(end_time - start_time):.4f} seconds')
        return result
    return wrapper
