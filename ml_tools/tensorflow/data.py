import glob
from typing import Dict, Union, List
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf


def load_dataset(
    data_details: Dict,
    batch_size: int,
    file_ffps: Union[List[str], np.ndarray] = None,
    data_dirs: Union[Path, List[Path]] = None,
    file_filter: str = "*.tfrecord",
    recursive: bool = False,
    shuffle_buffer: Union[int, None] = 1_000_000,
    n_open_files: int = 32,
    block_size: int = 256,
    n_parallel_calls: Union[int, None] = tf.data.experimental.AUTOTUNE
):
    """Performs the loading and parsing of a tensorflow dataset from
    the .tfrecord files in the given directory

    Parameters
    ----------

    data_details: dictionary
        Specifies how to parse the data,
        see https://www.tensorflow.org/api_docs/python/tf/io/parse_example?hl=en
        for more details
    batch_size: int
        Batch size, has to be done at this step as parsing of batches is
        much more efficient when using batches compared to single entries
    file_ffps: list of strings, optional
        The files that make up the database
        Either data_dirs (with file_filter) or file_ffps have to be given
    data_dirs: Path, list of Path, optional
        Directory that contains the .tfrecord files to use
        Either data_dirs (with file_filter) or file_ffps have to be given
    file_filter: str, optional
        The file filter to use when searching the
        specified directory for records
    recursive: bool, optional
        If specified then a recursive search is performed in each data_dir using the
        specified file_filter.
        No effect if file_ffps are given
    shuffle_buffer: int, optional
        Size of the shuffle buffer (in number of entries) to use,
        defaults to 1 Million
        If set to None, no shuffling is performed, however data is still loaded
        from multiple .tfrecord files at once (due to interleave) meaning that
        the data is still kinda "shuffled"
    n_open_files: int, optional
        How many .tfrecord files to read concurrently using the interleave
        function (https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
    block_size: int, optional
        Determines how many records are loaded from a .tfrecord file in one
        interleave cycle
        Default value should be fine, increasing it will make load times faster,
        however if the shuffle buffer is not appropriately sized then this may
        result in badly shuffled data
    n_parallel_calls: int, optional
        Number of parallel calls to use for interleave and parsing,
        default is tf.data.experimental.AUTOTUNE, which lets tensorflow determine the
        best number of calls to use.
        If sequential processing is required, then set this to None

    Returns
    -------
    tf.data.Dataset
        Note: The dataset will not return single training samples,
        but instead batches of batch_size!
    """
    # Either data_dirs (with file_filter) or file_ffps have to be given
    assert file_ffps is not None or data_dirs is not None

    if file_ffps is None:
        file_patterns = (
            [str(cur_dir / file_filter) for cur_dir in data_dirs]
            if isinstance(data_dirs, list)
            else [str(data_dirs / file_filter)]
        )
        file_ffps = np.concatenate(
            [
                glob.glob(cur_file_pattern, recursive=recursive)
                for cur_file_pattern in file_patterns
            ]
        )

    def _parse_fn(example_proto):
        parsed = tf.io.parse_example(example_proto, data_details)
        return parsed

    # 1) Create a dataset from the files, the shuffle option
    # means that the order of the files is shuffled every repeat (i.e. epoch) of the
    # dataset
    ds = tf.data.Dataset.from_tensor_slices(file_ffps).shuffle(
        len(file_ffps), reshuffle_each_iteration=True
    )

    # 2) Uses interleave cycle through the n_open_files .tfrecord files and feed one
    # one serialized sample into the shuffle buffer, opening new the next .tfrecord file
    # once one runs out of samples
    # 3) Shuffles all samples in the buffer and adding more into the buffer
    # as samples are removed
    # 4) Batch
    # 5) Parse each batch
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f),
        num_parallel_calls=n_parallel_calls,
        cycle_length=n_open_files,
        block_length=block_size,
        deterministic=False,
    )

    if shuffle_buffer is not None:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size).map(
        _parse_fn, num_parallel_calls=n_parallel_calls
    )

    return ds


def load_tfrecord(record_ffp: str, data_details: Dict, batch_size: int = 10_000):
    """
    Loads a single tfrecord file as a dictionary

    Parameters
    ----------
    record_ffp: string
        Path to the .tfrecord file
    data_details: dictionary
        Contains the data details required for loading
    batch_size: int, optional
        The batch size used for loading, this has to exceed the number samples
        in the .tfrecord file

    Returns
    -------
    dictionary:
        Data from the specified .tfrecord file
    """
    ds = tf.data.TFRecordDataset(filenames=[record_ffp])

    def _parse_fn(example_proto):
        parsed = tf.io.parse_example(example_proto, data_details)
        return parsed

    # Slight hack, just want to parse the whole record in one go,
    # not sure how to do this without batching...
    ds = ds.batch(batch_size).map(
        _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return next(ds.as_numpy_iterator())


def load_tfrecord_df(record_ffp: str, data_details: Dict, batch_size: int = 10_000):
    """
    Loads a single tfrecord file as a dataframe

    Parameters
    ----------
    record_ffp: string
        Path to the .tfrecord file
    data_details: dictionary
        Contains the data details required for loading
    batch_size: int, optional
        The batch size used for loading, this has to exceed the number samples
        in the .tfrecord file

    Returns
    -------
    dataframe:
        Data from the specified .tfrecord file
    """
    data = load_tfrecord(record_ffp, data_details, batch_size=batch_size)
    df = pd.DataFrame.from_dict(data)

    df.id = df.id.str.decode("UTF-8")
    df = df.set_index("id")

    return df
