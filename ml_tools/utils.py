import json
import datetime
import pickle
import math
from pathlib import Path
from typing import Dict, Any, Union, Sequence


import numpy as np
import yaml


class GenericObjJSONEncoder(json.JSONEncoder):
    """A generic JSON encoder that converts any unsupported object to a string"""

    def default(self, obj: Any) -> Any:
        """
        Attempt to encode the object, if it fails, convert it to a string

        Parameters
        ----------
        obj: Any
            The object to encode

        Returns
        -------
        Any
            The encoded object
        """
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as ex:
            return str(obj)


def round_sig(x: float, sig: int = 2):
    """
    Rounds the given number to the specified
    number of significant digits

    Parameters
    ----------
    x: float
        The number to round
    sig: int, optional
        The number of significant digits

    Returns
    -------
    float
        The rounded number

    References
    ----------
    [1] Stackoverflow: https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    """
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def create_run_id(include_seconds: bool = False) -> str:
    """
    Creates a run ID based on the month, day & time

    Parameters
    ----------
    include_seconds: bool, optional
        Whether to include the seconds in the ID

    Returns
    -------
    str
        The run ID
    """
    id = datetime.datetime.now().strftime(f"%m%d_%H%M{'%S' if include_seconds else ''}")
    return id


def write_to_json(data: Dict, ffp: Path, clobber: bool = False):
    """
    Writes the data to the specified file path in the json format

    Parameters
    ----------
    data: Dict
        The data to save
    ffp: Path
        The file path to save the data to
    clobber: bool, optional
        Whether to overwrite the file if it already exists
    """
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        json.dump(data, f, cls=GenericObjJSONEncoder, indent=4)


def write_to_yaml(data: Dict, ffp: Path, clobber: bool = False):
    """
    Writes the data to the specified file path in the yaml format

    Parameters
    ----------
    data: Dict
        The data to save
    ffp: Path
        The file path to save the data to
    clobber: bool, optional
        Whether to overwrite the file if it already exists
    """
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        yaml.safe_dump(data, f)


def write_to_txt(data: Sequence[Any], ffp: Path, clobber: bool = False):
    """
    Writes each entry in the list on a newline in a text file

    Parameters
    ----------
    data: Sequence[Any]
        The data to save
    ffp: Path
        The file path to save the data to
    clobber: bool, optional
        Whether to overwrite the file if it already exists
    """
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        f.writelines([f"{cur_line}\n" for cur_line in data])


def write_np_array(array: np.ndarray, ffp: Path, clobber: bool = False):
    """
    Saves the array in the numpy binary format .npy

    Parameters
    ----------
    array: np.ndarray
        The array to save
    ffp: Path
        The file path to save the array to
    clobber: bool, optional
        Whether to overwrite the file if it already exists
    """
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    np.save(str(ffp), array)


def write_pickle(obj: Any, ffp: Path, clobber: bool = False):
    """
    Saves the object as a pickle file

    Parameters
    ----------
    obj: Any
        The object to save
    ffp: Path
        The file path to save the object to
    clobber: bool, optional
        Whether to overwrite the file if it already exists
    """
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "wb") as f:
        pickle.dump(obj, f)


def load_picke(ffp: Union[Path, str]):
    """
    Loads the pickle file

    Parameters
    ----------
    ffp: Union[Path, str]
        The file path to load the data from

    Returns
    -------
    Any
        The loaded object
    """
    with open(ffp, "rb") as f:
        return pickle.load(f)


def load_txt(ffp: Union[Path, str]):
    """
    Loads a text file that has one entry per line

    Parameters
    ----------
    ffp: Union[Path, str]
        The file path to load the data from

    Returns
    -------
    List[str]
        The list of entries in the text file
    """
    with open(ffp, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_json(ffp: Union[Path, str]):
    """
    Loads a data from a json file

    Parameters
    ----------
    ffp: Union[Path, str]
        The file path to load the data from

    Returns
    -------
    Dict
        The loaded data
    """
    with open(ffp, "r") as f:
        return json.load(f)


def load_yaml(ffp: Union[Path, str]):
    """
    Loads data from a yaml file

    Parameters
    ----------
    ffp: Union[Path, str]
        The file path to load the data from

    Returns
    -------
    Dict
        The loaded data
    """
    with open(ffp, "r") as f:
        return yaml.safe_load(f)


def print_dict(data_dict: Dict):
    """
    Pretty prints a dictionary

    Parameters
    ----------
    data_dict: Dict
        The dictionary to print
    """
    print(json.dumps(data_dict, cls=GenericObjJSONEncoder, sort_keys=False, indent=4))


def normalise_timeseries(ts_values: np.ndarray[float]) -> np.ndarray[float]:
    """
    Normalises the timeseries to have zero mean and unit variance

    Parameters
    ----------
    ts_values: np.ndarray
        The timeseries values

    Returns
    -------
    np.ndarray
        The normalised timeseries values
    """
    return (ts_values - np.mean(ts_values, axis=1)[:, np.newaxis]) / np.std(
        ts_values, axis=1
    )[:, np.newaxis]


def compute_count_binned_trend(
    x_data: np.ndarray,
    y_data: np.ndarray,
    n_points_per_bin: int = None,
    n_bins: int = None,
    ignore_nans: bool = False,
):
    """
    Computes the trend for the given data, by splitting
    the data into bins such that the number of datapoints
    in each bin is equal

    Parameters
    -----------
    x_data: np.ndarray
        The x-axis data
    y_data: np.ndarray
        The y-axis data
    n_points_per_bin: int, optional
        The number of points in each bin
    n_bins: int, optional
        The number of bins to split the data into.
        If n_points_per_bin is specified, this parameter
        is ignored.
    ignore_nans: bool, optional
        Whether to ignore NaN values in the data
        when computing mean & standard deviation

    Returns
    --------
    bin_centers: np.ndarray
        Mid-point of the x-axis values in each bin
    bin_means: np.ndarray
        The mean of the y-axis values in each bin
    bin_stds: np.ndarray
        The standard deviation of the y-axis values in each bin
    """
    if not ignore_nans and np.isnan(y_data).any():
        print(f"Warning: NaN values detected in the data, and ignore_nans is False")

    if not n_points_per_bin and not n_bins:
        raise ValueError("Either n_points_per_bin or n_bins must be specified")

    n_points_per_bin = (
        n_points_per_bin if n_points_per_bin else math.ceil(len(x_data) / n_bins)
    )

    if n_bins is not None:
        assert n_bins == math.ceil(len(x_data) / n_points_per_bin)

    n_bins = math.ceil(len(x_data) / n_points_per_bin)

    # Sort the data based on the x-axis
    sort_indices = np.argsort(x_data)

    # Compute the trend for each bin
    bin_centers, bin_means, bin_stds = [], [], []
    for i in range(n_bins):
        start_idx = i * n_points_per_bin
        end_idx = (i + 1) * n_points_per_bin

        cur_x_data = x_data[sort_indices[start_idx:end_idx]]
        cur_y_data = y_data[sort_indices[start_idx:end_idx]]

        bin_centers.append(cur_x_data.min() + (cur_x_data.max() - cur_x_data.min()) / 2)

        bin_means.append(np.nanmean(cur_y_data) if ignore_nans else cur_y_data.mean())
        bin_stds.append(np.nanstd(cur_y_data) if ignore_nans else cur_y_data.std())

    return np.asarray(bin_centers), np.asarray(bin_means), np.asarray(bin_stds)


def compute_binned_trend(
    x_data: np.ndarray, y_data: np.ndarray, n_bins: int = 10, bins: np.ndarray = None
):
    """
    Computes the binned trend of the given data.

    Only one of n_bins or bins should be specified.

    Parameters
    ----------
    x_data: np.ndarray
        The x-axis data
    y_data: np.ndarray
        The y-axis data
    n_bins: int, optional
        The number of bins to split the data into
    bins: np.ndarray, optional
        The bin edges to use for splitting the data

    Returns
    -------
    bin_centers: np.ndarray
        The mid-point of the x-axis values in each bin
    bin_means: np.ndarray
        The mean of the y-axis values in each bin
    bin_stds: np.ndarray
        The standard deviation of the y-axis values in each bin
    """
    bins = np.linspace(x_data.min(), x_data.max(), n_bins + 1) if bins is None else bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    indices = np.digitize(x_data, bins)

    bin_means = np.asarray([y_data[indices == i].mean() for i in range(1, len(bins))])
    bin_stds = np.asarray([y_data[indices == i].std() for i in range(1, len(bins))])

    return bin_centers, bin_means, bin_stds
