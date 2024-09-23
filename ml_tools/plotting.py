import enum
from typing import Dict, Tuple, Sequence, NamedTuple, Union
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import utils


class LossPlotData(NamedTuple):
    """Loss plot data"""
    history: Dict
    label: str
    color: str


def plot_metrics(
    history: Dict,
    metric_keys: Sequence[str],
    metric_labels: Sequence[str] = None,
    ax: plt.Axes = None,
    y_lim: Tuple[float, float] = None,
    y_label: str = None,
    best_epoch: int = None,
):
    """
    Creates a loss plot for the given epoch history

    Parameters
    ----------
    history: Dict
        The epoch history dictionary
        Expected keys: {metric_keys}_train, {metric_keys}_val
    metric_keys: Sequence[str]
        The keys to plot from the history
    metric_labels: Sequence[str], optional
        The labels to use for the metrics
    ax: plt.Axes, optional
        The axes to use for plotting
        If None, a new figure is created
    y_lim: Tuple[float, float], optional
        The y-axis limits
    y_label:
        The y-axis label
    best_epoch: int, optional
        The best epoch to highlight in the plot

    Returns
    -------
    fig: plt.Figure
        The figure object
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(16, 10), dpi=200)
        ax = fig.add_subplot()

    epochs = np.arange(len(history[f"{metric_keys[0]}_train"]))

    colors = sns.color_palette("tab10", len(metric_keys))
    for ix, key in enumerate(metric_keys):
        cur_key = key if key in history.keys() else f"{key}_train"
        best_loss = (
            history[cur_key][best_epoch]
            if best_epoch is not None
            else np.min(history[cur_key])
        )
        label = (
            f"{cur_key if metric_labels is None else metric_labels[ix]} {best_loss:.2f}"
        )
        ax.plot(
            epochs,
            history[cur_key],
            linestyle="-",
            c=colors[ix],
            label=label,
        )
        if (cur_key := f"{key}_val") in history.keys():
            best_loss = (
                history[cur_key][best_epoch]
                if best_epoch is not None
                else np.min(history[cur_key])
            )
            val_label = f"{cur_key if metric_labels is None else metric_labels[ix]} {best_loss:.2f}"
            ax.plot(
                epochs,
                history[f"{key}_val"],
                linestyle="--",
                c=colors[ix],
                label=val_label if metric_labels is None else metric_labels[ix][1],
            )

    if best_epoch is not None:
        ax.axvline(best_epoch, c="r", linewidth=1.0, linestyle="--", label="Best epoch")

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" if y_label is None else y_label)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    return fig


def compare_loss(
    loss_plot_data: Sequence[LossPlotData], y_lim: Tuple[float, float] = None
):
    """
    Creates a plot for comparing multiple single-losses
    Not suitable for comparing losses from a multi-output model

    Parameters
    ----------
    loss_plot_data: Sequence[LossPlotData]
        The loss data to plot
    y_lim: Tuple[float, float], optional
        The y-axis limits

    Returns
    -------
    fig: plt.Figure
        The figure object
    """
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


class BinningMethod(enum.Enum):
    """The binning method for determining the trend line"""

    EqualXSized = "equal_x_sized"
    """Bins are created with equal x-axis sizes"""
    EqualCount = "equal_count"
    """Bins are created with equal data point counts"""


@dataclass
class ScatterOptions:
    x_axis: str
    """The x-axis column name"""
    y_axis: str
    """The y-axis column name"""

    x_min_use_qt: bool = False
    """Whether to use a quantile for the x-axis min"""
    x_max_use_qt: bool = False
    """Whether to use a quantile for the x-axis max"""
    x_min: Union[float, None] = None
    """The x-axis min value"""
    x_max: Union[float, None] = None
    """The x-axis max value"""

    y_min_use_qt: bool = False
    """Whether to use a quantile for the y-axis min"""
    y_max_use_qt: bool = False
    """Whether to use a quantile for the y-axis max"""
    y_min: Union[float, None] = None
    """The y-axis min value"""
    y_max: Union[float, None] = None
    """The y-axis max value"""

    use_fixed_color: bool = True
    """Whether to use a fixed color for the scatter points"""
    color: str = "blue"
    """The color of the scatter points"""

    color_axis: str = None
    """The column name to use for the color"""
    cmap: str = None
    """The colormap to use"""
    vmin_use_qt: bool = False
    """Whether to use a quantile for the min color"""
    vmax_use_qt: bool = False
    """Whether to use a quantile for the max color"""
    vmin: Union[float, None] = None
    """The min color value"""
    vmax: Union[float, None] = None
    """The max color value"""

    alpha: float = 1.0
    """The transparency of the scatter points"""
    marker_size: int = 5.0
    """The size of the scatter points"""

    binning_method: BinningMethod = BinningMethod.EqualXSized
    """The method to use for binning of the data for the trend line"""
    show_trend_mean_line: bool = True
    """Whether to show the mean trend line"""
    show_trend_std_line: bool = False
    """Whether to show the standard deviation of trend line"""
    trend_n_bins: int = 10
    """The number of bins to use for the trend line"""
    trend_n_data_points: int = None
    """The number of data points per trend line bin"""
    trend_line_style: str = "-"
    """The style of the trend line"""
    trend_line_width: float = 1.0
    """The width of the trend line"""
    trend_color: str = "blue"
    """The color of the trend line"""


def gen_scatter_trend_plot(df: pd.DataFrame, scatter_options: ScatterOptions):
    """
    Generates a scatter using the specified options
    For details on the options, see the ScatterOptions class

    Parameters
    ----------
    df: pd.DataFrame
        The data frame to use
    scatter_options: ScatterOptions
        The options to use for the plot

    Returns
    -------
    fig: plt.Figure
        The figure object
    ax: plt.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if scatter_options.use_fixed_color:
        ax.scatter(
            df[scatter_options.x_axis],
            df[scatter_options.y_axis],
            c=scatter_options.color,
            s=scatter_options.marker_size,
            alpha=scatter_options.alpha,
        )
    else:
        cm = ax.scatter(
            df[scatter_options.x_axis],
            df[scatter_options.y_axis],
            c=df[scatter_options.color_axis],
            cmap=scatter_options.cmap,
            s=scatter_options.marker_size,
            alpha=scatter_options.alpha,
            vmin=scatter_options.vmin,
            vmax=scatter_options.vmax,
        )
        plt.colorbar(cm, pad=0, label=scatter_options.color_axis)

    if scatter_options.show_trend_mean_line or scatter_options.show_trend_std_line:
        if scatter_options.binning_method == BinningMethod.EqualXSized:
            bin_centers, bin_means, bin_stds = utils.compute_binned_trend(
                df[scatter_options.x_axis].values,
                df[scatter_options.y_axis].values,
                n_bins=scatter_options.trend_n_bins,
            )
        elif scatter_options.binning_method == BinningMethod.EqualCount:
            bin_centers, bin_means, bin_stds = utils.compute_count_binned_trend(
                df[scatter_options.x_axis].values,
                df[scatter_options.y_axis].values,
                n_points_per_bin=scatter_options.trend_n_data_points,
                n_bins=scatter_options.trend_n_bins,
            )
        else:
            raise NotImplementedError()

        if scatter_options.show_trend_mean_line:
            ax.plot(
                bin_centers,
                bin_means,
                color=scatter_options.trend_color,
                linestyle=scatter_options.trend_line_style,
                marker=".",
            )
        if scatter_options.show_trend_std_line:
            ax.fill_between(
                bin_centers,
                bin_means - bin_stds,
                bin_means + bin_stds,
                color=scatter_options.trend_color,
                alpha=0.1,
            )

    ax.set_xlabel(scatter_options.x_axis)
    ax.set_ylabel(scatter_options.y_axis)

    ax.set_xlim(
        (
            (
                np.quantile(df[scatter_options.x_axis], scatter_options.x_min)
                if scatter_options.x_min_use_qt
                else scatter_options.x_min
            ),
            (
                np.quantile(df[scatter_options.x_axis], scatter_options.x_max)
                if scatter_options.x_max_use_qt
                else scatter_options.x_max
            ),
        )
    )
    ax.set_ylim(
        (
            (
                np.quantile(df[scatter_options.y_axis], scatter_options.y_min)
                if scatter_options.y_min_use_qt
                else scatter_options.y_min
            ),
            (
                np.quantile(df[scatter_options.y_axis], scatter_options.y_max)
                if scatter_options.y_max_use_qt
                else scatter_options.y_max
            ),
        )
    )

    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()
    return fig, ax
