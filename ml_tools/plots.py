from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_clusters(
    freq_values: np.ndarray, hvsr_values: np.ndarray, clusters: np.ndarray,
        y_lim: Tuple[float, float] = None,
        title: str = None
):
    """Plots each cluster in a subplot"""
    assert freq_values.size == hvsr_values.shape[1]

    # Plot the HVSR data
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    n_cluster = unique_clusters.size

    fig = plt.figure(figsize=(16, 10))

    # temp
    for cluster_ix, cur_cluster in enumerate(unique_clusters):
        ax = fig.add_subplot(n_cluster, 1, cluster_ix + 1)

        ax.set_xlim(freq_values.min(), freq_values.max())

        if cluster_ix == 0 and title is not None:
            ax.set_title(title)

        mask = clusters == cur_cluster
        cur_hvsr = hvsr_values[mask]

        for ix in range(cur_hvsr.shape[0]):
            plt.semilogx(
                freq_values,
                cur_hvsr[ix],
                alpha=1.0,
                c="gray",
                linewidth=0.5,
            )

        ax.text(
            0.99,
            0.95,
            f"Cluster {cur_cluster}, Count {cluster_counts[cluster_ix]}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_ylabel(f"HVSR")
        if cluster_ix < unique_clusters.size - 1:
            ax.xaxis.set_major_locator(ticker.NullLocator())

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    return fig


def plot_distance_matrix(dist_matrix_df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10))

    plt.imshow(dist_matrix_df.values, cmap="viridis")

    plt.colorbar()

    fig.tight_layout()
    return fig
