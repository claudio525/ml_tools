from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def precision_recall_threshold_plot(y_true: np.ndarray, y_est_prob: np.ndarray, output_ffp: Path):
    """Creates a precision-recall (y-axis) vs threshold (x-axis) plot"""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_est_prob)

    fig = plt.figure(figsize=(16, 12))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")

    plt.grid(linestyle="--", alpha=0.5)
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall")
    plt.legend()

    if output_ffp is not None:
        fig.savefig(output_ffp)
        plt.close()
    else:
        return fig