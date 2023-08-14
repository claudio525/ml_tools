from typing import Sequence

import numpy as np


def get_conv_out_sizes(
    input_length: int,
    kernel_sizes: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int] = None,
    max_pool_sizes: Sequence[int] = None
):
    """Computes the output sizes of the convolution layers"""
    padding = np.zeros_like(kernel_sizes) if padding is None else padding

    out_sizes = [input_length]
    for i in range(0, len(kernel_sizes)):
        cur_out_size = ((out_sizes[i] + 2 * padding[i] - (kernel_sizes[i] - 1) - 1) // stride[i]) + 1

        if max_pool_sizes is not None and i < len(max_pool_sizes):
            cur_out_size = int(np.floor((cur_out_size - (max_pool_sizes[i] - 1) - 1) / max_pool_sizes[i]) + 1)

        out_sizes.append(cur_out_size)

    return out_sizes[1:]


def get_trans_conv_out(
    input_length: int,
    kernel_sizes: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int] = None,
    output_padding: Sequence[int] = None,
):
    """Computes the output sizes of the transposed convolution layers"""
    padding = np.zeros_like(kernel_sizes) if padding is None else padding
    output_padding = (
        np.zeros_like(kernel_sizes) if output_padding is None else output_padding
    )

    out_sizes = [input_length]
    for i in range(0, len(kernel_sizes)):
        out_sizes.append(
            (out_sizes[i] - 1) * stride[i]
            - 2 * padding[i]
            + (kernel_sizes[i] - 1)
            + output_padding[i]
            + 1
        )

    return out_sizes[:-1]


def compute_decoder_out_padding(
    input_length: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: Sequence[int] = None,
):
    """
    Computes the output padding of the transposed convolution layers
    such that the output size matches the input size of the encoder
    """
    encoder_out_sizes = get_conv_out_sizes(input_length, kernel_sizes, strides, padding)

    out_padding = np.zeros_like(encoder_out_sizes)
    for i, cur_encoder_out in enumerate(np.flip(encoder_out_sizes)):
        # Has to be re-computed each time due to the flow on effect
        decoder_out_sizes = get_trans_conv_out(
            encoder_out_sizes[-1],
            np.flip(kernel_sizes),
            np.flip(strides),
            np.flip(padding) if padding is not None else None,
            output_padding=out_padding,
        )

        # Check for any differences
        if (cur_diff := np.abs(decoder_out_sizes[i] - cur_encoder_out)) > 0:
            out_padding[i - 1] = cur_diff

    return out_padding

def compute_same_conv_padding(input_length: int, kernel_size: int, stride: int = 1):
    return int(np.ceil(((input_length - 1) * stride + kernel_size - input_length) / 2))