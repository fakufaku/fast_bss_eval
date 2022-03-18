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

import math
from typing import Optional, Tuple, Union

import numpy as np
from .helpers import clamp


def optimal_symmetric_circulant_precond_column(
    col_toeplitz: np.ndarray,
) -> np.ndarray:
    """
    compute the first column of the circulant matrix closest to
    the Toeplitz matrix with provided first column in terms of the
    Froebenius norm
    """
    n = col_toeplitz.shape[-1]
    b = np.arange(1, n) / n
    w = col_toeplitz[..., 1:] * b[..., ::-1]
    col_circ = np.concatenate((col_toeplitz[..., :1], w + w[..., ::-1]), axis=-1)
    return col_circ


def optimal_nonsymmetric_circulant_precond_column(
    r: np.ndarray, n: Optional[int] = None, axis: Optional[int] = None
) -> np.ndarray:
    """
    compute the first column of the circulant matrix closest to
    the non-symmetric Toeplitz matrix with provided first column in terms of the
    Froebenius norm

    The input is assumed to be of shape (..., 2 * n_col) in order
    r = [s_0, s_1, ..., s_{n-1}, (s_{-n}) , s_{-n+1}, ..., s_{-1}]

    where the first row of the Toeplitz matrix is
    [s_0, ..., s_{n-1}]
    and the first column is
    [s_0, s_{-1}, ..., s_{-n+1}]
    """
    if axis is not None:
        r = np.moveaxis(r, axis, -1)

    if n is None:
        n = (r.shape[-1] + 1) // 2

    b = np.arange(1, n) / n

    s1 = r[..., n - 1 : 0 : -1]
    s2 = r[..., :-n:-1]

    w = b * s1 + (1.0 - b) * s2

    c = np.concatenate((r[..., :1], w), axis=-1)

    if axis is not None:
        c = np.moveaxis(c, -1, axis)

    return c


class CirculantPreconditionerOperator:
    """
    Optimal circulant pre-conditioner operator for a symmetric topelitz matrix
    """

    def __init__(self, toeplitz_col: np.ndarray):
        col_precond = optimal_symmetric_circulant_precond_column(toeplitz_col)
        C = np.fft.rfft(col_precond, axis=-1)
        # complex pointwise inverse
        self.C = C.conj() / clamp(C.real**2 + C.imag**2, min=1e-6)
        self.n = col_precond.shape[-1]

        self._shape = col_precond.shape[:-1] + (self.n, self.n)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __matmul__(self, lhs: np.ndarray) -> np.ndarray:

        if lhs.ndim == self.ndim:
            assert lhs.shape[-2] == self.shape[-1]
            return np.fft.irfft(
                self.C[..., None] * np.fft.rfft(lhs, n=self.n, axis=-2),
                n=self.n,
                axis=-2,
            )

        else:
            raise ValueError(
                "Dimension mismatch between operators with "
                f"shapes {self.shape} and {lhs.shape}"
            )


class SymmetricToeplitzOperator:
    def __init__(self, col: np.ndarray):
        """
        col: numpy.ndarray, (..., n_chan_ref, filter_length)
        """
        n_col = col.shape[-1]
        # n_fft = 2 ** math.ceil(math.log2(2 * n_col - 1))
        n_fft = 2 * n_col
        pad_len = n_fft - 2 * n_col + 1
        pad_shape = col.shape[:-1] + (pad_len,)
        circ_col = np.concatenate(
            (col, np.zeros_like(col, shape=pad_shape), col[..., :0:-1]), axis=-1
        )
        self.Cforward = np.fft.rfft(circ_col, axis=-1)

        self._shape = col.shape + (col.shape[-1],)
        self.n_col = n_col
        self.n_fft = n_fft

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __matmul__(self, lhs: np.ndarray) -> np.ndarray:
        if lhs.ndim == self.ndim:
            assert lhs.shape[-2] == self.shape[-1]
            Y = np.fft.rfft(lhs, n=self.n_fft, axis=-2)
            y = np.fft.irfft(Y * self.Cforward[..., None], n=self.n_fft, axis=-2)
            return y[..., : self.n_col, :]

        else:
            raise ValueError(
                "Dimension mismatch between operators with "
                f"shapes {self.shape} and {lhs.shape}"
            )


class BlockCirculantPreconditionerOperator:
    """
    Optimal circulant pre-conditioner operator for a symmetric topelitz matrix
    """

    def __init__(self, toeplitz_col: np.ndarray):
        """
        toeplitz_col: numpy.ndarray, (..., 2 * filter_length, n_channels, n_channels)
        """

        col_precond = optimal_nonsymmetric_circulant_precond_column(
            toeplitz_col, axis=-3
        )
        C = np.fft.rfft(col_precond, axis=-3)

        # complex pointwise inverse
        self.C = np.linalg.inv(C)
        self.n = col_precond.shape[-3]

        self._shape = col_precond.shape[:-1] + (
            self.n * self.C.shape[-2],
            self.n * self.C.shape[-1],
        )

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __matmul__(self, lhs: np.ndarray) -> np.ndarray:

        assert lhs.shape[-2] == self.shape[-1], (
            "Dimension mismatch between operators with "
            f"shapes {self.shape} and {lhs.shape}"
        )

        lhs = lhs.reshape(lhs.shape[:-2] + (-1, self.C.shape[-1]) + lhs.shape[-1:])
        lhs = np.fft.rfft(lhs, n=self.n, axis=-3)
        prod = np.einsum("...rc,...cm->...rm", self.C, lhs)
        y = np.fft.irfft(prod, n=self.n, axis=-3)
        y = y.reshape(lhs.shape[:-3] + (-1,) + lhs.shape[-1:])

        return y


class BlockToeplitzOperator:
    """
    Operator implementing fast matrix multiplication by a block Toeplitz matrix.
    Each block of the matrix is a symmetric Toeplitz matrix.
    """

    def __init__(self, col: np.ndarray):
        """
        Parameters
        ----------
        col: numpy.ndarray, (..., 2 * n_toeplitz, n_block_rows, n_block_cols)
            The reduced representation  of the block Toeplitz matrix.
            The last dimension contains the concatenated first column and first row
            of the Toeplitz matrix.
            We can thus directly apply the FFT
        """
        n_col = col.shape[-3] // 2

        n_fft = 2 ** math.ceil(math.log2(2 * n_col))
        # n_fft = 2 * n_col

        pad_len = n_fft - 2 * n_col + 1
        pad_shape = col.shape[:-3] + (pad_len,) + col.shape[-2:]
        circ_col = np.concatenate(
            (
                col[..., :1, :, :],
                col[..., :-n_col:-1, :, :],
                np.zeros_like(col, shape=pad_shape),
                col[..., n_col - 1 : 0 : -1, :, :],
            ),
            axis=-3,
        )
        self.Cforward = np.fft.rfft(circ_col, axis=-3)

        self._n_block_rows = col.shape[-2]
        self._n_block_cols = col.shape[-1]
        self._n_toeplitz = n_col

        self.n_fft = circ_col.shape[-3]

        self._shape = col.shape[:-3] + (
            self._n_block_rows * n_col,
            self._n_block_cols * n_col,
        )
        self.n_fft = n_fft

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __matmul__(self, lhs: np.ndarray) -> np.ndarray:
        assert lhs.shape[-2] == self.shape[-1], (
            "Dimension mismatch between operators with "
            f"shapes {self.shape} and {lhs.shape}"
        )
        # lhs.shape = (..., n_toeplitz, n_block_cols, n_lhs)
        lhs = lhs.reshape(lhs.shape[:-2] + (-1, self._n_block_cols) + lhs.shape[-1:])
        Y = np.fft.rfft(lhs, n=self.n_fft, axis=-3)
        prod = np.einsum("...rc,...cm->...rm", self.Cforward, Y)
        y = np.fft.irfft(prod, n=self.n_fft, axis=-3)

        # truncate output
        y = y[..., : self._n_toeplitz, :, :]

        # reshape as vector
        y = y.reshape(y.shape[:-3] + (-1,) + y.shape[-1:])

        return y


class IdentityOperator:
    def __matmul__(self, lhs: np.ndarray) -> np.ndarray:
        return lhs


def inner_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...cd,...cd->...d", a.conj(), b)


def conjugate_gradient(
    A: Union[np.ndarray, BlockToeplitzOperator, SymmetricToeplitzOperator],
    b: np.ndarray,
    x: Optional[np.ndarray] = None,
    n_iter: Optional[int] = None,
    precond: Optional[
        Union[
            np.ndarray,
            BlockCirculantPreconditionerOperator,
            CirculantPreconditionerOperator,
        ]
    ] = None,
    verbose: Optional[bool] = False,
    x_true: Optional[np.ndarray] = None,
) -> np.ndarray:

    # assert hasattr(A, "__matmul__")

    add_axis = A.ndim - 1 == b.ndim
    if add_axis:
        b = b[..., None]

    assert len(A.shape) == len(b.shape) and A.shape[-1] == b.shape[-2]

    if x is None:
        x = np.zeros_like(b)

    if n_iter is None:
        n_iter = b.shape[-1]

    if precond is None:
        precond = IdentityOperator()

    r = b - A @ x
    z = precond @ r
    p = z

    rsold = inner_prod(r, z)

    if x_true is not None:
        if add_axis:
            x_true = x_true[..., None]
        errors = [np.mean(np.linalg.norm(x - x_true, axis=-2))]

    for epoch in range(n_iter):
        Ap = A @ p
        pAp = inner_prod(p, Ap)
        alpha = rsold / clamp(pAp, min=1e-6)
        x = x + alpha[..., None, :] * p
        if epoch + 1 % 5 == 0:
            r = b - A @ x
        else:
            r = r - alpha[..., None, :] * Ap
        z = precond @ r

        rsnew = inner_prod(r, z)
        if verbose:
            print(rsnew.mean())

        beta = rsnew / clamp(rsold, min=1e-6)
        p = z + beta[..., None, :] * p
        rsold = rsnew

        if x_true is not None:
            errors.append(np.mean(np.linalg.norm(x - x_true, axis=-2)))

    if add_axis:
        x = x[..., 0]

    if x_true is not None:
        return x, errors
    else:
        return x


def toeplitz_conjugate_gradient(
    acf: np.ndarray,
    xcorr: np.ndarray,
    x: Optional[np.ndarray] = None,
    n_iter: Optional[int] = None,
    x_true: Optional[np.ndarray] = None,
) -> np.ndarray:

    # prepare pre-conditioner
    precond = CirculantPreconditionerOperator(acf)

    # prepare forward operator
    forward = SymmetricToeplitzOperator(acf)

    # run conjugate gradient
    return conjugate_gradient(
        forward, xcorr, n_iter=n_iter, precond=precond, x=x, x_true=x_true
    )


def block_toeplitz_conjugate_gradient(
    acf: np.ndarray,
    xcorr: np.ndarray,
    x: Optional[np.ndarray] = None,
    n_iter: Optional[int] = None,
    x_true: Optional[np.ndarray] = None,
) -> np.ndarray:

    # prepare pre-conditioner
    precond = BlockCirculantPreconditionerOperator(acf)
    # precond = None

    # prepare forward operator
    forward = BlockToeplitzOperator(acf)

    # run conjugate gradient
    return conjugate_gradient(
        forward, xcorr, n_iter=n_iter, precond=precond, x=x, x_true=x_true
    )
