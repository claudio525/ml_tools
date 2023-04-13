import tensorflow as tf

from ml_tools.tensorflow import hidden_layers

HIDDEN_LAYER_MAPPING = {
    "relu_bn": hidden_layers.relu_bn,
    "relu": hidden_layers.relu,
    "selu": hidden_layers.selu,
    "elu": hidden_layers.elu,
    "tanh": hidden_layers.tanh,
    "elu_mc_dropout": hidden_layers.elu_mc_dropout,
    "selu_mc_dropout": hidden_layers.selu_mc_dropout,
}

OPTIMIZER_MAPPING = {
    "adam": tf.optimizers.Adam,
    "sgd": tf.optimizers.SGD,
}

def get_hidden_layer_fn(name: str):
    return HIDDEN_LAYER_MAPPING[name]

def get_optimizer(name: str, **kwargs):
    return OPTIMIZER_MAPPING[name](**kwargs)
