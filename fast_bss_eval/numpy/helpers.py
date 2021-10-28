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

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def clamp(x, min=None, max=None):

    if min is not None:
        x = np.maximum(x, min)

    if max is not None:
        x = np.minimum(x, max)

    return x


def _remove_mean(x: np.ndarray, axis: Optional[int] = -1) -> np.ndarray:
    return x - x.mean(axis=axis, keepdims=True)


def _normalize(
    x: np.ndarray, eps: Optional[float] = 1e-6, axis: Optional[int] = None
) -> np.ndarray:
    x = x / clamp(np.linalg.norm(x, axis=axis, keepdims=True), min=eps)
    return x


def _db_clamp_eps(db_max: float) -> float:
    e = 10.0 ** (-db_max / 10.0)
    eps = e / (1.0 + e)
    return eps


def _coherence_to_neg_sdr(
    coh: np.ndarray, clamp_db: Optional[float] = 60.0
) -> np.ndarray:

    if clamp_db is not None:
        # clamp within desired decibel range
        eps = _db_clamp_eps(clamp_db)
        coh = clamp(coh, min=eps, max=1 - eps)

    ratio = (1 - coh) / coh

    # apply the SDR mapping
    return 10.0 * np.log10(ratio)


def _solve_permutation(
    target_loss_matrix: np.ndarray,
    *args,
    return_perm=False,
) -> Tuple[np.ndarray]:
    """
    Solve the permutation in numpy for now
    """

    loss_mat = target_loss_matrix  # more consice name

    b_shape = loss_mat.shape[:-2]
    n_chan_ref, n_chan_est = loss_mat.shape[-2:]
    n_chan_out = min(n_chan_ref, n_chan_est)

    if n_chan_ref > n_chan_est:
        loss_mat = loss_mat.swapaxes(-2, -1)
        args = list(args)
        for i, arg in enumerate(args):
            args[i] = arg.swapaxes(-2, -1)

    loss_out = np.zeros_like(loss_mat, shape=b_shape + (n_chan_out,))
    args_out = [np.zeros_like(arg, shape=b_shape + (n_chan_out,)) for arg in args]

    p_opts = np.zeros(b_shape + (n_chan_out,), dtype=np.int64)
    for m in np.ndindex(b_shape):
        # linear sum assignment tries to *maximize* the sum, so we add a minus sign
        # because we are supposed to *minimize* losses
        dum, p_opt = _linear_sum_assignment_with_inf(loss_mat[m])
        loss_out[m] = loss_mat[m + (dum, p_opt)]
        for i, arg in enumerate(args):
            args_out[i][m] = arg[m + (dum, p_opt)]
        p_opts[m] = p_opt

    if return_perm:
        return (loss_out,) + tuple(args_out) + (p_opts,)
    else:
        if len(args_out) == 0:
            return loss_out
        else:
            return (loss_out,) + tuple(args_out)


def _linear_sum_assignment_with_inf(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the permutation problem efficiently via the linear sum
    assignment problem.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    This implementation was proposed by @louisabraham in
    https://github.com/scipy/scipy/issues/6900
    to handle infinite entries in the cost matrix.
    """
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        cost_matrix = cost_matrix.copy()
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)
