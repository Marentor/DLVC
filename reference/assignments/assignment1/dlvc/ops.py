import random
from typing import List, Callable

import numpy as np

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]


def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op


def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.ravel()

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = sample + val
        return sample

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        sample = sample * val
        return sample

    return op


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(sample, (2, 0, 1))

    return op


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            sample = np.fliplr(sample)
        return sample

    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if (len(sample) + pad) < sz:
            raise ValueError
        if pad > 0:
            sample = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), pad_mode)
        w, h = sample.shape[:2]
        x, y = np.random.randint(h - sz), np.random.randint(w - sz)
        sample = sample[y:y + sz, x:x + sz]
        sample = np.resize(sample, (32, 32, 3))
        return sample

    return op

    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html will be helpful
