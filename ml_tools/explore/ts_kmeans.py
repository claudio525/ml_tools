"""
Runs K-Means on the TMP HVSRs from Chris

As expected, doesnt really work all that well
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn import metrics
from tslearn import neighbors
from tslearn.utils import to_time_series
from scipy.spatial import distance

from tslearn import preprocessing
from tslearn import clustering

hvsr_data_ffp = Path(
    "/Users/claudy/dev/work/data/site_response/HVSR_meanCurve_TempAndMAM.txt"
)
resp_data_ffp = Path(
    "/Users/claudy/dev/work/data/site_response/HVSR_meanCurve_HuttGrid_RefTestForSMS.txt"
)

tmp_hvsr_df = pd.read_csv(hvsr_data_ffp, index_col=0, delim_whitespace=True)
sms_hvsr_df = pd.read_csv(resp_data_ffp, index_col=0, delim_whitespace=True)

assert np.all(tmp_hvsr_df.columns == sms_hvsr_df.columns)

freq_values = np.asarray(
    [float(c.rsplit("_", maxsplit=1)[0]) for c in tmp_hvsr_df.columns]
)

mask = freq_values >= 0.1
freq_values = freq_values[mask]

tmp_hvsr_df = tmp_hvsr_df.loc[:, mask]
sms_hvsr_df = sms_hvsr_df.loc[:, mask]

freq_values = freq_values[::2]
tmp_hvsr_df = tmp_hvsr_df.iloc[:, ::2]
sms_hvsr_df = sms_hvsr_df.iloc[:, ::2]


# Preprocess
# pre_hvsr = preprocessing.TimeSeriesScalerMeanVariance().fit_transform(
#     tmp_hvsr_df.values
# )

pre_hvsr = np.log(tmp_hvsr_df.values)

# Run Clustering
km = clustering.TimeSeriesKMeans(
    5,
    n_init=10,
    metric="dtw",
    metric_params=dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=2),
    n_jobs=-1
)
clusters = km.fit_predict(pre_hvsr)

# Plot the HVSR data
unique_clusters = np.unique(clusters)
n_cluster = unique_clusters.size

fig = plt.figure(figsize=(16, 10))

# temp
ax1 = None
for ix, cur_cluster in enumerate(unique_clusters):
    ax = fig.add_subplot(n_cluster, 1, ix + 1, sharex=ax1)

    if ix == 0:
        ax1 = ax
        ax.set_xlim(freq_values.min(), freq_values.max())

    mask = clusters == cur_cluster
    cur_hvsr = tmp_hvsr_df.values[mask]

    for ix in range(cur_hvsr.shape[0]):
        plt.semilogx(
            freq_values,
            cur_hvsr[ix],
            alpha=0.75,
            c="gray",
            linewidth=0.5,
        )

    ax.set_ylim(0.0, 10.0)
    ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax.set_ylabel(f"HVSR")
    ax.yaxis.set_ticks([0, 2, 4, 6, 8])

plt.xlabel(f"Frequency")

plt.legend()

plt.tight_layout()
fig.subplots_adjust(hspace=0.0)


plt.show()

print(f"wtf")
