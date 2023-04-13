from typing import Dict, Tuple, Sequence, NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LossPlotData(NamedTuple):
    history: Dict
    label: str
    color: str

def plot_loss(
    history: Dict,
    ax: plt.Axes = None,
    y_lim: Tuple[float, float] = None,
    y_label: str = None,
    multi_keys: Sequence[str] = None,
    plot_train: bool = True,
    plot_val: bool = True,
):
    """Creates a loss plot for the given training history"""
    assert plot_val or plot_train

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(16, 10), dpi=200)
        ax = fig.add_subplot()

    epochs = np.arange(len(history["loss"]))

    if multi_keys is None:
        if plot_train:
            ax.plot(
                epochs,
                history["loss"],
                linestyle="-",
                c="k",
                label=f"Training: {np.min(history['loss']):.2f}",
            )
        if plot_val:
            ax.plot(
                epochs,
                history["val_loss"],
                linestyle="--",
                c="k",
                label=f"Validation: {np.min(history['val_loss']):.2f}",
            )

    # Multi-output loss
    if multi_keys is not None:
        colors = sns.color_palette("tab10", int((len(history) - 2) / 2))
        for cur_key, cur_color in zip(multi_keys, colors):
            cur_t_key, cur_v_key = f"{cur_key}_loss", f"val_{cur_key}_loss"
            if plot_train:
                ax.plot(
                    epochs,
                    history[cur_t_key],
                    linestyle="-",
                    c=cur_color,
                    label=f"Training - {cur_key}: {np.min(history[cur_t_key]):.2f}",
                )
            if plot_val:
                ax.plot(
                    epochs,
                    history[cur_v_key],
                    linestyle="--",
                    c=cur_color,
                    label=f"Validation - {cur_key}: {np.min(history[cur_v_key]):.2f}",
                )

    if y_lim is not None:
        ax.ylim(y_lim)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if y_label is None else y_label)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    return fig





def compare_loss(
    loss_plot_data: Sequence[LossPlotData], y_lim: Tuple[float, float] = None
):
    """Creates a plot for comparing multiple single-losses, i.e. not for
    comparing losses from a multi-output model"""
    fig = plt.figure(figsize=(16, 10), dpi=200)

    for cur_data in loss_plot_data:
        cur_epochs = np.arange(len(cur_data.history["loss"]))
        plt.plot(
            cur_epochs,
            cur_data.history["loss"],
            linestyle="-",
            c=cur_data.color,
            label=f"{cur_data.label}, Training: {np.min(cur_data.history['loss']):.2f}",
        )
        plt.plot(
            cur_epochs,
            cur_data.history["val_loss"],
            linestyle="--",
            c=cur_data.color,
            label=f"{cur_data.label}, Validation: {np.min(cur_data.history['val_loss'])::.2f}",
        )

    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.legend()

    return fig
