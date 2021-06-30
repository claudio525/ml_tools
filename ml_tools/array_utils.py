import numpy as np

def find_nearest(array: np.ndarray, value: float):
    ix = (np.abs(array - value)).argmin()
    return ix
