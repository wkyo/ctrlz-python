# -*- coding: utf-8 -*-
from typing import (
    Any,
    Callable,
    Sequence,
)

import numpy as np


def array_distance_split(array: np.ndarray, distance: int = 5):
    """Split 1-D array by element distance.

    Let the maximum distance between elements be `s`, for continuous elements
    `a, b, c, d`. If `|a-b| < s`, `|b-c| > s` and `|c-d| < s`, the elements
    `a, b` are divided into one partition, and the elements`c, d` are divided
    into another partition.

    Parameters
    ----------
    array : numpy.ndarray
        An 1-D array in ascending order.
    distance : int, optional
        Element spacing, by default 5.

    Returns
    -------
    indices : list
        A list of indices, such as `[3, 6, 9]`.

    Examples
    --------
    >>> a = [0,1,2,9,10,12,17,20,24,30]
    >>> array_distance_split(a, distance=5)
    [3, 6, 9]
    """
    indices = []
    last_num = None
    for idx, num in enumerate(array):
        if last_num is not None and num - last_num >= distance:
            indices.append(idx)
        last_num = num
    return indices


def nonzero_partions(array: np.ndarray, return_array: bool = False):
    """Get continuos nonzero partions from an 1-D array.

    Parameters
    ----------
    array : numpy.ndarray
        An 1-D array.
    return_array : bool, optional
        Return the arrays consisting of nonzero elements, by default False.

    Returns
    -------
    partions : list
        A list of partions, each partition consisting of a start index and an
        stop index, such as [(1, 3), (5, 6), (9, 11)].
    arrays : list
        A list of nonzero arrays.

    Examples
    --------
    >>> a = [0,1,2,0,0,5,0,0,0,9,10,0,0]
    >>> nonzero_partions(a)
    [(1, 3), (5, 6), (9, 11)]
    """
    partions = []
    left_idx = -1
    for idx, value in enumerate(array):
        if value > 0 and left_idx < 0:  # Rising edge
            left_idx = idx
        elif value == 0 and left_idx >= 0:  # Falling edge
            partions.append((left_idx, idx))
            left_idx = -1
    if left_idx >= 0:  # handle remained partions
        partions.append((left_idx, len(array)))

    if return_array:
        arrays = [array[j:k] for j, k in partions]
        return partions, arrays
    return partions


def max_pool_1d(
    array: np.ndarray,
    ksize: int = 3,
    stride: int = 1,
    padding: int = 0,
    value: int = 0,
):
    """1-D maximum pooling.

    Output size is :math:`\lfloor \frac{L_\mathtt{in} + 2 * \mathtt{padding} \
        - \mathtt{ksize}}{\mathtt{stride}} \rfloor  + 1`.

    If you want to get an output with the same size as input array, stride
    should be 1, padding should be equal to `ksize // 2`, which ksize must be
    odd.

    Parameters
    ----------
    array : numpy.ndarray
        An 1-D array.
    ksize : int, optional
        Kernel size, by default 3.
    stride : int, optional
        Strde steps, by default 1.
    padding : int, optional
        Padding with N elements on both sides, by default 0.
    value : int, optional
        Default padding value, by default 0.

    Returns
    -------
    numpy.ndarray
        An 1-D array

    Examples
    --------
    >>> a = np.arange(10)
    >>> max_pool_1d(a, ksize=2, stride=2).tolist()
    [1, 3, 5, 7, 9]
    >>> max_pool_1d(a, ksize=3, stride=1, padding=1).tolist()
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    >>> max_pool_1d(a, ksize=5, stride=1, padding=2).tolist()
    [2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    """
    L_in = len(array)
    L_out = (L_in + 2 * padding - ksize) // stride + 1

    output = np.zeros(L_out, dtype=array.dtype)

    for idx_out in range(L_out):
        idx_in_left = idx_out * stride - padding
        idx_in_right = idx_in_left + ksize

        if idx_in_left < 0 < idx_in_right:
            v = np.max(array[:idx_in_right])
            v = max(v, value)
        elif idx_in_left >= 0 and idx_in_right <= L_in:
            v = np.max(array[idx_in_left:idx_in_right])
        elif idx_in_left < L_in < idx_in_right:
            v = np.max(array[idx_in_left:])
            v = max(v, value)
        else:
            v = value

        output[idx_out] = v

    return output
