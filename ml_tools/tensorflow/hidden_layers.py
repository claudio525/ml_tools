from typing import Dict, Union

from tensorflow import keras
from tensorflow.keras import layers

class MCDropout(keras.layers.Dropout):
    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs, **kwargs):
        return super(MCAlphaDropout, self).call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


def elu_mc_dropout(input: layers.Layer, n_units: int, dropout: float = None):
    x = layers.Dense(
        units=n_units, activation="elu", kernel_initializer="glorot_uniform"
    )(input)
    if dropout is not None and dropout > 0.0:
        x = MCDropout(rate=dropout)(x)

    return x


def relu_bn(input: layers.Layer, n_units: int, l2: float = None):
    reg = keras.regularizers.L2(l2) if l2 is not None else None
    x = layers.Dense(units=n_units, activation="relu", kernel_regularizer=reg)(input)
    x = layers.BatchNormalization()(x)

    return x


def relu(input: layers.Layer, n_units: int, dropout: float = None, l2: float = None, name: str = None):
    reg = keras.regularizers.L2(l2) if l2 is not None else None

    x = layers.Dense(units=n_units, activation="relu", kernel_regularizer=reg, name=name)(input)
    if dropout is not None and dropout > 0.0:
        x = layers.Dropout(rate=dropout)(x)

    return x


def elu(input: layers.Layer, n_units: int, dropout: float = None, l2: float = None, name: str = None):
    reg = keras.regularizers.L2(l2) if l2 is not None else None

    x = layers.Dense(units=n_units, activation="elu", kernel_regularizer=reg, name=name)(input)
    if dropout is not None and dropout > 0.0:
        x = layers.Dropout(rate=dropout)(x)

    return x

def tanh(input: layers.Layer, n_units: int, dropout: float = None, l2: float = None, name: str = None):
    reg = keras.regularizers.L2(l2) if l2 is not None else None

    x = layers.Dense(units=n_units, activation="tanh", kernel_regularizer=reg, name=name)(input)
    if dropout is not None and dropout > 0.0:
        x = layers.Dropout(rate=dropout)(x)

    return x

def selu(
    input: layers.Layer,
    n_units: int,
    dropout: Union[float, None] = None,
    l2: Union[float, None] = None,
):
    x = layers.Dense(
        units=n_units,
        activation="selu",
        kernel_initializer="lecun_normal",
        kernel_regularizer=keras.regularizers.l2(l2) if l2 is not None else None,
    )(input)
    if dropout is not None and dropout > 0.0:
        x = layers.AlphaDropout(dropout)(x)

    return x

def selu_mc_dropout(input: layers.Layer, n_units: int, dropout: float = None):
    x = layers.Dense(
        units=n_units, activation="selu", kernel_initializer="lecun_normal"
    )(input)
    if dropout is not None and dropout > 0.0:
        x = MCAlphaDropout(dropout)(x)

    return x


def cnn1_mc_dropout_pool(
    input: layers.Layer,
    filters: int,
    kernel_size: int,
    cnn_config: Dict,
    dropout: float = None,
    pool_size: int = 2,
):
    x = layers.Conv1D(filters, kernel_size, **cnn_config)(input)
    if dropout is not None:
        x = MCSpatialDropout1D(rate=dropout)(x)
    if pool_size is not None:
        x = keras.layers.MaxPooling1D(pool_size)(x)

    return x


def bi_lstm(input: layers.Layer, n_units: int, **lstm_config):
    x = keras.layers.Bidirectional(
        layer=keras.layers.LSTM(units=n_units, **lstm_config)
    )(input)

    return x



