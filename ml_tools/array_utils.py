from typing import Sequence

import numpy as np
import pandas as pd


def find_nearest(array: np.ndarray[float], value: float) -> int:
    """
    Returns the index of the entry that is
    closest to the specified value.

    This assumes that the array is sorted!

    Parameters
    ----------
    array: array of floats
        The array to search
    value: float
        The value to search for

    Returns
    -------
    ix: int
        The index of the nearest value

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.5, 3.8, 4.2, 5.9, 7.1])
    >>> find_nearest(arr, 3.9)
    2
    >>> find_nearest(arr, 5.0)
    3
    >>> find_nearest(arr, 7.1)
    5
    """
    ix = (np.abs(array - value)).argmin()
    return ix


def find_nearest_larger(array: np.ndarray[float], value: float) -> int:
    """
    Returns the index of the entry that is
    closest to the specified value. If no exact match
    is found the nearest larger value is returned.

    This assumes that the array is sorted!

    Parameters
    ----------
    array: array of floats
        The array to search
    value: float
        The value to search for

    Returns
    -------
    ix: int
        The index of the nearest (larger) value

    Raises
    ------
    ValueError
        If the value is larger than the largest value in the array

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.5, 3.8, 4.2, 5.9, 7.1])
    >>> find_nearest_larger(arr, 4.0)
    3
    >>> find_nearest_larger(arr, 5.0)
    4
    >>> find_nearest_larger(arr, 7.1)
    5
    """
    ix = find_nearest(array, value)

    if array[ix] == value or array[ix] > value:
        return ix
    else:
        if ix == array.size - 1:
            raise ValueError("Value is larger than the largest value in the array")
        return ix + 1


def find_nearest_smaller(array: np.ndarray[float], value: float) -> int:
    """
    Returns the index of the entry that is
    closest to the specified value. If no exact match
    is found the nearest larger value is returned.

    This assumes that the array is sorted!

    Parameters
    ----------
    array: array of floats
        The array to search
    value: float
        The value to search for

    Returns
    -------
    ix: int
        The index of the nearest (smaller) value

    Raises
    ------
    ValueError
        If the value is smaller than the smallest value in the array

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.5, 3.8, 4.2, 5.9, 7.1])
    >>> find_nearest_smaller(arr, 4.0)
    2
    >>> find_nearest_smaller(arr, 5.0)
    3
    >>> find_nearest_smaller(arr, 1.0)
    0
    """
    ix = find_nearest(array, value)

    if array[ix] == value or array[ix] < value:
        return ix
    else:
        if ix == 0:
            raise ValueError("Value is smaller than the smallest value in the array")
        return ix - 1
    
def find_nearest_smaller_vec(array: np.ndarray[float], values: np.ndarray[float]) -> np.ndarray[int]:
    """
    Returns the indices of the entries that are
    closest to the specified values. If no exact match
    is found the nearest smaller value is returned.

    This assumes that the array is sorted!

    Parameters
    ----------
    array: array of floats
        The array to search
    values: array of floats
        The values to search for

    Returns
    -------
    ix: np.ndarray[int]
        The indices of the nearest (smaller) values

    Raises
    ------
    ValueError
        If any value is smaller than the smallest value in the array

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.5, 3.8, 4.2, 5.9, 7.1])
    >>> values = np.array([4.0, 5.0, 1.0])
    >>> find_nearest_smaller_vec(arr, values)
    array([2, 3, 0])
    """
    if len(array.shape) != 1:
        raise ValueError("Input array must be 1D")
    if len(values.shape) != 1:
        raise ValueError("Input values must be 1D")
    
    tmp = np.tile((np.arange(array.size)), (values.shape[0], 1))
    return np.argmax(np.where(array <= values[:, None], tmp, -1), axis=1)


def pandas_isin(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """
    This is the same as a np.isin,
    however it is significantly faster for large arrays

    Parameters
    ----------
    array_1: array of any type
    array_2: array of any type

    Returns
    -------
    np.ndarray
        The boolean array

    References
    ----------
    [1] Stackoverflow: https://stackoverflow.com/questions/15939748/check-if-each-element-in-a-numpy-array-is-in-another-array

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> arr2 = np.array([1, 3, 5])
    >>> pandas_isin(arr, arr2)
    array([ True, False,  True, False,  True])
    """
    return pd.Index(pd.unique(array_2)).get_indexer(array_1) >= 0


def numpy_str_join(sep: str, *arrays: str | Sequence[str]) -> np.ndarray[str]:
    """
    Joins multiple string arrays together using the specified separator.
    Also support joining of string values, or a combination of both.

    Parameters
    ----------
    sep: str
        The separator to use
    arrays: string value or string arrays
        The arrays (or string values) to join together

    Returns
    -------
    np.ndarray
        The joined string array

    Examples
    --------
    Example 1: Joining two arrays
    >>> numpy_str_join("_", ["a", "b"], ["c", "d"])
    array(['a_c', 'b_d'], dtype='<U3')

    Example 2: Combination of string value and arrays
    >>> arr1 = np.array(["a", "b"])
    >>> numpy_str_join("_", arr1, "c", ["d", "e"])
    array(['a_c_d', 'b_c_e'], dtype='<U5')
    """
    result = arrays[0]
    for cur_array in arrays[1:]:
        result = np.char.add(result, sep)
        result = np.char.add(result, cur_array)

    return result
