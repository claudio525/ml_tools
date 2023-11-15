import numpy as np
import pandas as pd


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



def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """This is the same as a np.isin,
    however is significantly faster for large arrays

    https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0

def numpy_str_join(sep: str, *arrays: np.ndarray):
    """Join multiple numpy arrays of strings"""
    result = arrays[0]
    for cur_array in arrays[1:]:
        result = np.char.add(result, sep)
        result = np.char.add(result, cur_array)

    return result
