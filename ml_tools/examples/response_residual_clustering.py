from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial import distance

from ml_tools import clustering
from ml_tools import plots
from ml_tools import dist_calc

# Config Options
# Modify these as needed

# Minimum and maximum period to consider
# for clustering
# Set to None to use min/max of the data
min_period, max_period = None, None

# Window size for smoothing
# Averages (median) the specified number of
# data points into a single value
# Increasing this will reduce the effect of
# smaller peaks in the residual, see
# first plot for effect of this
win_size = 5


# Specifies the Dynamic Time Warping
# radius constraint
DTW_radius = 5

# Linkage method
# Has to be one of:
#   complete, single, average, ward
# See here for details:
# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
linkage_method = "ward"

# Number of clusters
n_clusters = 6

def load_res_pSA_from_csv(im_data_ffp: Path):
    """Loads the residual PSA values from a csv file
    (I.e. Data from Ayushi)
    """
    im_df = pd.read_csv(im_data_ffp, index_col=0).T

    # Only interested in pSA
    pSA_columns = np.asarray([cur_col for cur_col in im_df.columns if cur_col.startswith("pSA")])
    im_df = im_df.loc[:, pSA_columns]

    # Get the pSA period values
    pSA_period_values = [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_columns]
    im_df.columns = pSA_period_values

    # Ensure columns were already sorted
    assert np.all(np.sort(pSA_period_values) == pSA_period_values)

    # Drop any with nan
    im_df = im_df.dropna()

    return im_df

# Load the IM data
res_data_ffp = Path(
    "/Users/claudy/dev/work/data/hvsr_analysis/ayushi_data/PJSreStationBiased_sim.csv"
)
res_df = load_res_pSA_from_csv(res_data_ffp)
X = res_df.copy()

# Filter periods
if min_period or max_period is not None:
    min_period = X.columns.min() if min_period is None else min_period
    max_period = X.columns.max() if max_period is None else max_period

    mask = (X.columns.values >= min_period) & (X.columns.values <= max_period)
    X = X.loc[:, mask]
    res_df = res_df.loc[:, mask]

# Determine needed shape
n_periods = int(np.ceil(X.shape[1] / win_size) * win_size)

# Smooth
X_smooth = np.pad(X, ((0, 0), (0, n_periods - X.shape[1])), "edge")
period_values = np.median(
    np.pad(X.columns.values, (0, n_periods - X.shape[1]), "edge").reshape(-1, win_size),
    axis=1,
)
X_smooth = X_smooth.reshape((X.shape[0], -1, win_size)).mean(axis=2)
X = pd.DataFrame(index=X.index, data=X_smooth, columns=period_values)


# Plot smoothed residual spectra
fig = plt.figure(figsize=(16, 10))
color_palette = sns.color_palette("Paired", 10)
for ix in range(10):
    plt.semilogx(
        res_df.columns, res_df.iloc[ix, :].values, linewidth=0.75, c=color_palette[ix]
    )
    plt.semilogx(
        period_values,
        X.iloc[ix, :],
        linewidth=0.75,
        c=color_palette[ix],
        linestyle="--",
    )
plt.xlim(res_df.columns.min(), res_df.columns.max())
plt.xlabel(f"Period")
plt.ylabel(f"Residual - pSA")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()


# Compute the distance matrix
dist_matrix_df = dist_calc.compute_dtw_dist_matrix(
    X, dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=DTW_radius)
)

# Run hierachical clustering
Z = clustering.compute_hierarchical_linkage_matrix(
    distance.squareform(dist_matrix_df.values), method=linkage_method
)
cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion="maxclust")

# Plot the dendrogram
fig = plt.figure(figsize=(12, 9))
dn = hierarchy.dendrogram(Z)
plt.title(f"Dendrogram")
plt.tight_layout()

# Plot the clusters
fig = plots.plot_clusters(
    res_df.columns, res_df.values, cluster_labels, title="Residuals", y_lim=(-1.0, 1.0)
)

# Show the plots
plt.show()
