import numpy as np


def find_nearest(array: np.ndarray, value: float):
    """Returns the index of the entry that is
    closest to the specified value"""
    ix = (np.abs(array - value)).argmin()
    return ix


def find_nearest_larger(array: np.ndarray, value: float):
    """Returns the index of the entry that is
    closest to the specified value. If no exact match
    is found the nearest larger value is returned."""
    ix = find_nearest(array, value)

    if array[ix] == value or array[ix] > value:
        return ix
    else:
        return ix + 1


def find_nearest_smaller(array: np.ndarray, value: float):
    """Returns the index of the entry that is
    closest to the specified value. If no exact match
    is found the nearest larger value is returned."""
    ix = find_nearest(array, value)

    if array[ix] == value or array[ix] < value:
        return ix
    else:
        return ix - 1
