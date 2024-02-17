"""
custom_timeit.py

Overwrites the python timeit template to allow capturing return values
from functions that are being timed

"""


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
