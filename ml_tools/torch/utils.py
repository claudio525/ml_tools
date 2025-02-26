import torch.nn.functional as F
import torch.nn as nn



def get_act_fn(act_fn: str | None) -> callable:
    """Returns the activation function based on the input string."""
    if act_fn == "relu":
        return F.relu
    elif act_fn == "tanh":
        return F.tanh
    elif act_fn == "sigmoid":
        return F.sigmoid
    elif act_fn == "softmax":
        return F.softmax
    elif act_fn == "leaky_relu":
        return F.leaky_relu
    elif act_fn == "elu":
        return F.elu
    elif act_fn == "selu":
        return F.selu
    elif act_fn == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Activation function {act_fn} not supported.")


def get_act_fn_layer(act_fn: str | None) -> nn.Module:
    """Returns the activation function layer based on the input string."""
    if act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "tanh":
        return nn.Tanh()
    elif act_fn == "sigmoid":
        return nn.Sigmoid()
    elif act_fn == "softmax":
        return nn.Softmax(dim=1)
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "elu":
        return nn.ELU()
    elif act_fn == "selu":
        return nn.SELU()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Activation function {act_fn} not supported.")