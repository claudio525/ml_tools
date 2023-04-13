from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hdbscan
from sklearn import decomposition

from hvsr_analysis import dist_calc
from hvsr_analysis import utils
from hvsr_analysis import plots

def load_hsvr_from_txt(hvsr_ffp: Path, min_freq: float = None):
    """Loads the HVSR values from a txt file
    (I.e. Data from Chris)
    """
    hvsr_df = pd.read_csv(hvsr_ffp, index_col=0, delim_whitespace=True)

    # Get the corresponding frequency values
    freq_values = np.asarray(
        [float(c.rsplit("_", maxsplit=1)[0]) for c in hvsr_df.columns]
    )

    # Sanity checks
    assert freq_values.size == hvsr_df.columns.size
    assert np.all(np.sort(freq_values) == freq_values)

    if min_freq is not None:
        mask = freq_values >= min_freq
        freq_values = freq_values[mask]
        hvsr_df = hvsr_df.loc[:, mask]

    return freq_values, hvsr_df


# Load the HVSR data
hvsr_data_ffp = Path(
    "/Users/claudy/dev/work/data/site_response/HVSR_meanCurve_TempAndMAM.txt"
)
freq_values, hvsr_df = utils.load_hsvr_from_txt(hvsr_data_ffp, min_freq=0.1)

# Normalise the HVSR series
hvsr_df_norm = hvsr_df.copy(deep=True)
hvsr_df_norm.loc[:, :] = utils.norm_ts(hvsr_df.values)

# Compute the distance matrix
dist_matrix_df = dist_calc.compute_dtw_dist_matrix(
    hvsr_df_norm, dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
)

## Visualize distance matrix
# plots.plot_distance_matrix(dist_matrix_df)
# plt.show()

## Run PCA
# pca = decomposition.PCA(2).fit(dist_matrix_df.values)
# X_PCA = pca.transform(dist_matrix_df.values)
#
# fig = plt.figure(figsize=(16, 10))
#
# plt.scatter(X_PCA[:, 0], X_PCA[:, 1])
#
# plt.title(f"PCA")
# plt.xlabel(f"X2")
# plt.ylabel(f"X1")
# plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
# plt.tight_layout()
#
# plt.show()


## Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(dist_matrix_df.values)

fig = plots.plot_clusters(freq_values, hvsr_df_norm.values, cluster_labels, title="Normalized HVSR")

fig = plots.plot_clusters(freq_values, hvsr_df.values, cluster_labels, title="Original HVSR", y_lim=(0, 10.0))
plt.show()

print(f"wtf")
