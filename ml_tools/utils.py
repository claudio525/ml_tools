import json
import datetime
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Sequence

import tensorflow as tf
import pandas as pd
import numpy as np
import yaml
import wandb

from . import hidden_layers

HIDDEN_LAYER_MAPPING = {
    "relu_bn": hidden_layers.relu_bn,
    "relu_dropout": hidden_layers.relu_dropout,
    "selu_dropout": hidden_layers.selu,
    "elu_mc_dropout": hidden_layers.elu_mc_dropout,
    "selu_mc_dropout": hidden_layers.selu_mc_dropout,
}

def get_hidden_layer_fn(name: str):
    return HIDDEN_LAYER_MAPPING[name]

class GenericObjJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as ex:
            return str(obj)


def create_run_id(include_seconds: bool = False) -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime(f"%m%d_%H%M{'%S' if include_seconds else ''}")
    return id


def write_to_json(data: Dict, ffp: Path, clobber: bool = False):
    """Writes the data to the specified file path in the json format"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        json.dump(data, f, cls=GenericObjJSONEncoder, indent=4)

def write_to_yaml(data: Dict, ffp: Path, clobber: bool = False):
    """Writes the data to the specified file path in the yaml format"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        yaml.safe_dump(data, f)

def write_to_txt(data: Sequence[Any], ffp: Path, clobber: bool = False):
    """Writes each entry in the list on a newline in a text file"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        f.writelines([f"{cur_line}\n" for cur_line in data])

def write_np_array(array: np.ndarray, ffp: Path, clobber: bool = False):
    """Saves the array in the numpy binary format .npy"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    np.save(str(ffp), array)

def write_pickle(obj: Any, ffp: Path, clobber: bool = False):
    """Saves the object as a pickle file"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "wb") as f:
        pickle.dump(obj, f)


def load_picke(ffp: Union[Path, str]):
    """Loads the pickle file"""
    with open(ffp, "rb") as f:
        return pickle.load(f)

def load_txt(ffp: Union[Path, str]):
    """Loads a text file that has one entry per line"""
    with open(ffp, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_json(ffp: Union[Path, str]):
    """Loads a data from a json file"""
    with open(ffp, "r") as f:
        return json.load(f)


def load_yaml(ffp: Union[Path, str]):
    """Loads data from a yaml file"""
    with open(ffp, "r") as f:
        return yaml.safe_load(f)


def print_dict(data_dict: Dict):
    """Pretty prints a dictionary"""
    print(json.dumps(data_dict, cls=GenericObjJSONEncoder, sort_keys=False, indent=4))


def save_print_data(output_dir: Path, wandb_save: bool = True, **kwargs):
    """Save and if specified prints the given data
    To print, set the value to a tuple, with the 2nd entry a
    boolean that indicates whether to print or not,
    i.e. hyperparams=(hyperparams_dict, True) will save and print
    the hyperparameter dictionary
    """
    for key, value in kwargs.items():
        if isinstance(value, tuple) and value[1] is True:
            _print(key, value[0])
            _save(output_dir, key, value[0], wandb_save)
        else:
            _save(output_dir, key, value, wandb_save)


def _print(key: str, data: Union[pd.DataFrame, pd.Series, dict]):
    print(f"{key}:")
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        print(data)
    elif isinstance(data, dict):
        print_dict(data)


def _save(
    output_dir: Path,
    key: str,
    data: Union[pd.DataFrame, pd.Series, dict],
    wandb_save: bool,
):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        out_ffp = output_dir / f"{key}.csv"
        data.to_csv(out_ffp)
    elif isinstance(data, dict):
        out_ffp = output_dir / f"{key}.json"
        write_to_json(data, out_ffp)
    elif isinstance(data, list):
        out_ffp = output_dir / f"{key}.txt"
        write_to_txt(data, out_ffp)
    elif isinstance(data, np.ndarray):
        out_ffp = output_dir / f"{key}.npy"
        write_np_array(data, out_ffp)
    elif isinstance(data, tf.keras.Model):
        out_ffp = output_dir / f"{key}.png"
        tf.keras.utils.plot_model(
            data,
            to_file=str(out_ffp),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
        )
    else:
        raise NotImplementedError()

    if wandb_save:
        wandb.save(str(out_ffp))




