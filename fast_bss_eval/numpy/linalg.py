# Copyright 2021 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def hankel_view(x: np.ndarray, n_rows: int) -> np.ndarray:
    """return a view of x as a Hankel matrix"""
    n_cols = x.shape[-1] - n_rows + 1
    return np.lib.stride_tricks.as_strided(
        x, shape=x.shape[:-1] + (n_rows, n_cols), strides=x.strides + (x.strides[-1],)
    )


def toeplitz(x):
    """
    make symmetric toeplitz matrix
    """
    two_x = np.concatenate((x[..., :0:-1], x), axis=-1)
    H = hankel_view(two_x, n_rows=x.shape[-1])
    T = H[..., ::-1, :]
    return T


def block_toeplitz(x, n=None):
    """
    Create a block matrix where the blocks have a Toeplitz structure
    """
    top, rows, cols = x.shape[-3:]

    if n is None:
        n = top // 2

    out = np.zeros_like(x, shape=x.shape[:-3] + (n, rows, n, cols))

    for r in range(x.shape[-2]):
        for c in range(x.shape[-1]):
            vec = np.concatenate((x[..., -n + 1 :, r, c], x[..., :n, r, c]), axis=-1)
            T = hankel_view(vec, n)
            out[..., :, r, :, c] = T[..., ::-1, :]

    out = out.reshape(out.shape[:-4] + (n * rows, n * cols))

    return out
