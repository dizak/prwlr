__version__ = '0.0.4'
__author__ = 'Dariusz Izak IBB PAS'


__all__ = [
    "apis",
    "databases",
    "errors",
    "network",
    "profiles",
    "stats",
    "utils"
]

try:
    import pandas as _pd
    import numpy as _np
    import subprocess as _sp
    from prwlr.core import *
except ImportError:
    print('Could not import dependencies. Ignore if running setup.')
