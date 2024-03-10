"""
custom_timeit.py

Overwrites the python timeit template to allow capturing return values
from functions that are being timed

"""


from __future__ import annotations

import statistics
import timeit

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


def custom_timeit(func, n_repeat, n_loop, *args):
    """Custom timeit, run func in a loop n_loop times and repeat for n_repeat times.
    Return the output of the function, the mean and std of the runtimes"""

    runtimes = []
    return_values = []

    timer = timeit.Timer(lambda: func(*args))

    # repeat loop n_repeat times
    for _ in range(n_repeat):
        total_runtime, return_value = timer.timeit(number=n_loop)
        runtimes.append(total_runtime / n_loop)
        return_values.append(return_value)

    # calculate standard deviation
    mean_runtime = sum(runtimes) / n_repeat
    std_runtime = statistics.stdev(runtimes)

    # assume deterministic function and return only first element of return values
    return mean_runtime, std_runtime, return_values[0]
