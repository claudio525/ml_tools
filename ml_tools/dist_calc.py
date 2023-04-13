from typing import Dict

import numpy as np
import pandas as pd
from tslearn import metrics


def compute_dtw_dist_matrix(
    series_df: pd.DataFrame, dtw_params: Dict = None
) -> pd.DataFrame:
    """
    Computes the DTW distance matrix for the given values

    Parameters
    ----------
    series_df: array of floats
        The dataframe of series
        for which to compute
        the distance matrix

        Expected shape: [N, K] where
        N = Number of series
        K = Number of steps
    dtw_params: Dictionary
        Parameters that are passed on to
        tslearn.metrics.dtw

    Returns
    -------
    pd.DataFrame:
        Distance matrix
        Shape: [N, N]
    """
    dtw_params = dict() if dtw_params is None else dtw_params

    N = series_df.shape[0]
    dist_matrix = np.full((N, N), fill_value=np.nan)
    for i in range(N):
        for k in range(i, N):
            dist_matrix[k, i] = dist_matrix[i, k] = metrics.dtw(
                series_df.iloc[i], series_df.iloc[k], **dtw_params
            )

    # Sanity checks
    assert np.allclose(dist_matrix, dist_matrix.T)
    assert np.all(~np.isnan(dist_matrix))

    return pd.DataFrame(
        data=dist_matrix, index=series_df.index, columns=series_df.index
    )
