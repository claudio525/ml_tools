from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tslearn import neighbors
from tslearn.utils import to_time_series


hvsr_data_ffp = Path(
    "/Users/claudy/dev/work/data/site_response/HVSR_meanCurve_TempAndMAM.txt"
)
resp_data_ffp = Path(
    "/Users/claudy/dev/work/data/site_response/HVSR_meanCurve_HuttGrid_RefTestForSMS.txt"
)

# Load the data
tmp_hvsr_df = pd.read_csv(hvsr_data_ffp, index_col=0, delim_whitespace=True)
sms_hvsr_df = pd.read_csv(resp_data_ffp, index_col=0, delim_whitespace=True)

# Compute the logged-hvsr
# Not used, just left in here in case you want to
# try with logged values
ln_tmp_hvsr_df = np.log(tmp_hvsr_df)
ln_sms_hvsr_df = np.log(sms_hvsr_df)

# Sanity check
assert np.all(tmp_hvsr_df.columns == sms_hvsr_df.columns)

# Get the frequency values
freq_values = np.asarray(
    [float(c.rsplit("_", maxsplit=1)[0]) for c in tmp_hvsr_df.columns]
)

# Drop all frequencies above 0.1
mask = freq_values >= 0.1
freq_values = freq_values[mask]

tmp_hvsr_df = tmp_hvsr_df.loc[:, mask]
ln_tmp_hvsr_df = ln_tmp_hvsr_df.loc[:, mask]

sms_hvsr_df = sms_hvsr_df.loc[:, mask]
ln_sms_hvsr_df = ln_sms_hvsr_df.loc[:, mask]


# Select 10 random HVSRs from the tmp dataset
rand_ind = np.random.randint(0, ln_tmp_hvsr_df.shape[0], 10)
hvsrs = tmp_hvsr_df.iloc[rand_ind, :].values

# Run nearest neighbour for the different distance metrics

# DTW
dtw_knn = neighbors.KNeighborsTimeSeries(n_neighbors=1, metric="dtw").fit(
    to_time_series(sms_hvsr_df.values)
)
dtw_dist, dtw_ind = dtw_knn.kneighbors(to_time_series(hvsrs))

# Convert resulting indices and distances to 1d arrays
dtw_ind = np.ravel(dtw_ind)
dtw_dist = np.ravel(dtw_dist)

# DTW(R=2)
dtw2_knn = neighbors.KNeighborsTimeSeries(
    n_neighbors=1,
    metric="dtw",
    metric_params=dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=2),
).fit(to_time_series(sms_hvsr_df.values))
dtw2_dist, dtw2_ind = dtw2_knn.kneighbors(to_time_series(hvsrs))
dtw2_ind = np.ravel(dtw2_ind)
dtw2_dist = np.ravel(dtw2_dist)

# Euclidean Distance
ec_knn = neighbors.KNeighborsTimeSeries(n_neighbors=1, metric="euclidean").fit(
    to_time_series(sms_hvsr_df.values)
)
ec_dist, ec_ind = ec_knn.kneighbors(to_time_series(hvsrs))
ec_ind = np.ravel(ec_ind)
ec_dist = np.ravel(ec_dist)

# Plotting
for ix, cur_hvsr in enumerate(hvsrs):
    fig = plt.figure(figsize=(16, 10))

    plt.semilogx(freq_values, cur_hvsr, c="k", linewidth=1.0, label="X")
    plt.semilogx(
        freq_values,
        sms_hvsr_df.iloc[dtw_ind[ix], :].values,
        c="r",
        linewidth=1.0,
        label=f"DTW: {dtw_dist[ix]:.2f}",
    )
    plt.semilogx(
        freq_values,
        sms_hvsr_df.iloc[dtw2_ind[ix], :].values,
        c="r",
        linestyle="--",
        linewidth=1.0,
        label=f"DTW(R=2): {dtw2_dist[ix]:.2f}",
    )
    plt.semilogx(
        freq_values,
        sms_hvsr_df.iloc[ec_ind[ix], :].values,
        c="b",
        linestyle="-",
        linewidth=1.0,
        label=f"Euclidean: {ec_dist[ix]:.2f}",
    )

    for _, cur_row in sms_hvsr_df.iterrows():
        plt.semilogx(freq_values, cur_row.values, alpha=0.5, c="gray", linewidth=0.5)

    plt.xlabel(f"Frequency")
    plt.ylabel(f"HVSR")
    plt.xlim(freq_values.min(), freq_values.max())
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.show()
