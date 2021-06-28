"""Contains shared utility functions that operate on data."""

from operator import attrgetter
import numpy as np

def get_uint_ctype(integer):
    """Gets smallest possible uint representation of provided integer.
    Raises ValueError if invalid value provided (negative or above uint64)."""
    for dtype in np.typecodes["UnsignedInteger"]:
        min_val, max_val = attrgetter('min', 'max')(np.iinfo(np.dtype(dtype)))
        if min_val <= integer <= max_val:
            return dtype
    raise ValueError("Cannot parse {} as a uint64".format(integer))
