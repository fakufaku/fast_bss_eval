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
from typing import Optional, Tuple

import torch

from .cgd import block_toeplitz_conjugate_gradient, toeplitz_conjugate_gradient
from .helpers import (
    _coherence_to_neg_sdr,
    _normalize,
    _remove_mean,
    _solve_permutation,
)
from .linalg import toeplitz, block_toeplitz


def square_cosine_metrics_length_one_filter(
    ref: torch.Tensor,
    est: torch.Tensor,
    zero_mean: Optional[bool] = False,
    pairwise: Optional[bool] = True,
    load_diag: Optional[float] = None,
    with_coh_sar: Optional[bool] = True,
) -> Tuple[torch.Tensor, ...]:
    """
    Special case of computing cosine metrics with filter length of 1
    """

    if zero_mean:
        ref = _remove_mean(ref, dim=-1)
        est = _remove_mean(est, dim=-1)

    ref = _normalize(ref, dim=-1)
    est = _normalize(est, dim=-1)

    if pairwise or with_coh_sar:
        xcorr = torch.einsum("...cn,...dn->...cd", ref, est)

    if pairwise:
        coh_sdr = xcorr
    else:
        coh_sdr = torch.einsum("...n,...n->...", ref, est)
    coh_sdr = torch.square(coh_sdr)

    if with_coh_sar:
        acm = torch.einsum("...cn,...dn->...cd", ref, ref)
        if load_diag is not None:
            acm = acm + torch.eye(acm.shape[-1]) * load_diag
        sol = torch.linalg.solve(acm, xcorr)
        coh_sar = torch.einsum("...lc,...lc->...c", xcorr, sol)

        if pairwise:
            coh_sdr, coh_sar = torch.broadcast_tensors(coh_sdr, coh_sar[..., None, :])

        return coh_sdr, coh_sar

    else:
        return coh_sdr


def compute_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    length: Optional[int] = None,
    pairwise: Optional[bool] = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the auto correlation function of x and its cross-correlation with y.
    This function is specialized for when only the SDR is needed.
    In this case, only the auto-correlation of each source is required.

    Parameters
    ----------
    x: torch.Tensor, (..., n_chan_x, n_samples)
        Usually the reference signals
    y: torch.Tensor, (..., n_chan_y, n_samples)
        Usually the estimated signals
    length: int, optional
        The length at which to truncate the statistics
    pairwise: bool, optional
        When this flag is true, statistics are computed for all
        pairs of source/estimate. This only affects the cross-correlation.

    Returns
    -------
    acf: (..., n_chan_x, length)
        The auto-correlation functions of x
    xcorr: torch.Tensor
        The cross-correlation functions of x and y.
        The shape is (..., n_chan_x, length, n_chan_y) when pairwise is True, and
        (..., n_chan_x, length) when pairwise is False.
    """
    if length is None:
        length = x.shape[-1]

    max_len = max(x.shape[-1], y.shape[-1])
    n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + max_len - 1))

    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    Y = torch.fft.rfft(y, n=n_fft, dim=-1)

    # autocorrelation function
    acf = torch.fft.irfft(X.real**2 + X.imag**2, n=n_fft)

    # cross-correlation
    if pairwise:
        XY = torch.einsum("...cn,...dn->...cnd", X.conj(), Y)
        xcorr = torch.fft.irfft(XY, n=n_fft, dim=-2)
        return acf[..., :length], xcorr[..., :length, :]

    else:
        XY = X.conj() * Y
        xcorr = torch.fft.irfft(XY, n=n_fft, dim=-1)
        return acf[..., :length], xcorr[..., :length]


def compute_stats_2(
    x: torch.Tensor,
    y: torch.Tensor,
    length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the auto correlation function of x and its cross-correlation with y.
    This function computes the full auto-correlation matrix for all reference signals.
    This is only needed when both the SDR and the SIR have to be computed.

    Parameters
    ----------
    x: torch.Tensor, (..., n_chan_x, n_samples)
        Usually the reference signals
    y: torch.Tensor, (..., n_chan_y, n_samples)
        Usually the estimated signals
    length: int, optional
        The length at which to truncate the statistics

    Returns
    -------
    acf: (..., length, n_chan_x, n_chan_y)
        The auto-correlation functions of x
    xcorr: torch.Tensor, (..., length, n_chan_x, n_chan_y)
        The cross-correlation functions of y
    """
    if length is None:
        length = x.shape[-1]

    max_len = max(x.shape[-1], y.shape[-1])
    n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + max_len - 1))

    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    Y = torch.fft.rfft(y, n=n_fft, dim=-1)

    # autocorrelation function
    prod = torch.einsum("...cn,...dn->...ncd", X, X.conj())
    acf = torch.fft.irfft(prod, n=n_fft, dim=-3)
    acf = torch.cat([acf[..., :length, :, :], acf[..., -length:, :, :]], dim=-3)

    # cross-correlation
    XY = torch.einsum("...cn,...dn->...ncd", X.conj(), Y)
    xcorr = torch.fft.irfft(XY, n=n_fft, dim=-3)
    xcorr = xcorr[..., :length, :, :]

    return acf, xcorr


def sdr_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
    pairwise: Optional[bool] = False,
) -> torch.Tensor:
    """
    Compute the negative signal-to-distortion ratio (SDR).

    The order of est/ref follows the convention of pytorch loss functions
    (i.e., est first).

    Parameters
    ----------
    est: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    ref: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
    filter_length: int, optional
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter: int, optional
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean: bool, optional
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db: bool, optional
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    load_diag: float, optional
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    pairwise: bool, optional
        If set to True, the metrics are computed for every est/ref signals pair (default: False).
        When set to False, it is expected that ``n_channels_ref == n_channels_est``.

    Returns
    -------
    neg_cisdr: torch.Tensor
        The negative SDR of the input signal. The returned tensor has shape
        (..., n_channels_est) if ``pairwise == False`` and
        (..., n_channels_ref, n_channels_est) if ``pairwise == True``.
    """

    if filter_length == 1:
        coh = square_cosine_metrics_length_one_filter(
            ref,
            est,
            zero_mean=zero_mean,
            load_diag=None,
            pairwise=pairwise,
            with_coh_sar=False,
        )

    else:

        if zero_mean:
            est = _remove_mean(est)
            ref = _remove_mean(ref)

        # normalize along time-axis
        est = _normalize(est, dim=-1)
        ref = _normalize(ref, dim=-1)

        # compute auto-correlation and cross-correlation
        acf, xcorr = compute_stats(ref, est, length=filter_length, pairwise=pairwise)

        if load_diag is not None:
            # the diagonal factor of the Toeplitz matrix is the first
            # coefficient of the acf
            acf[..., 0] += load_diag

        # solve for the optimal filter
        if use_cg_iter is None:
            # regular matrix solver
            R_mat = toeplitz(acf)
            sol = torch.linalg.solve(R_mat, xcorr)
        else:
            # use preconditioned conjugate gradient
            sol = toeplitz_conjugate_gradient(acf, xcorr, n_iter=use_cg_iter)

        # compute the coherence
        if pairwise:
            coh = torch.einsum("...lc,...lc->...c", xcorr, sol)
        else:
            coh = torch.einsum("...l,...l->...", xcorr, sol)

    # transform to decibels
    neg_sdr = _coherence_to_neg_sdr(coh, clamp_db=clamp_db)

    return neg_sdr


def sdr_pit_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
) -> torch.Tensor:
    return sdr(
        ref,
        est,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        load_diag=load_diag,
        change_sign=True,
        return_perm=False,
    )


def pairwise_sdr_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the negative signal-to-distortion ratio (SDR).

    This function computes the SDR for all pairs of est/ref signals.

    The order of est/ref follows the convention of pytorch loss functions
    (i.e., est first).

    Parameters
    ----------
    est: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    ref: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
    filter_length: int, optional
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter: int, optional
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean: bool, optional
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db: bool, optional
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    load_diag: float, optional
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero

    Returns
    -------
    neg_cisdr: torch.Tensor, (..., n_channels_ref, n_channels_est)
        The negative SDR of the input signal
    """
    return sdr_loss(
        est,
        ref,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        load_diag=load_diag,
        pairwise=True,
    )


def sdr(
    ref: torch.Tensor,
    est: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
    return_perm: Optional[bool] = False,
    change_sign: Optional[bool] = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute the signal-to-distortion ratio (SDR) only.

    This function computes the SDR for all pairs of est/ref signals and finds the
    permutation maximizing the SDR.

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    est: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
    filter_length: int, optional
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter: int, optional
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean: bool, optional
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db: bool, optional
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    load_diag: float, optional
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    return_perm: bool, optional
        If set to True, the optimal permutation of the estimated signals is
        also returned (default: ``False``)
    change_sign: bool, optional
        If set to True, the sign is flipped and the negative SDR is returned
        (default: ``False``)

    Returns
    -------
    cisdr: torch.Tensor, (..., n_channels_est)
        The SDR of the input signal
    perm: torch.Tensor, (..., n_channels_est), optional
        The index of the corresponding reference signal.
        Only returned if ``return_perm == True``
    """

    neg_sdr = sdr_loss(
        est,
        ref,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        pairwise=True,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        load_diag=load_diag,
    )

    neg_sdr, perm = _solve_permutation(neg_sdr, return_perm=True)

    sdr = neg_sdr if change_sign else -neg_sdr

    if return_perm:
        return sdr, perm
    else:
        return sdr


def square_cosine_metrics(
    ref: torch.Tensor,
    est: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    pairwise: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the square cosines of the angles between

    1. the estimate and the subspaces of shifts of each of the reference signals
    2. the estimate and the subspace of shifts of all reference signals

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    est: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
    filter_length: int, optional
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter: int, optional
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean: bool, optional
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    pairwise: bool, optional
        When this flag is true, statistics are computed for all
        pairs of source/estimate. This only affects the cross-correlation.
    load_diag: float, optional
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero

    Returns
    -------
    neg_cisdr: torch.Tensor, (..., n_channels_ref, n_channels_est)
        The negative SDR of the input signal
    """

    # In this case, we can compute things more efficiently
    if filter_length == 1:
        return square_cosine_metrics_length_one_filter(
            ref,
            est,
            zero_mean=zero_mean,
            pairwise=pairwise,
            load_diag=load_diag,
            with_coh_sar=True,
        )

    if zero_mean:
        est = _remove_mean(est)
        ref = _remove_mean(ref)

    # normalize along time-axis
    est = _normalize(est, dim=-1)
    ref = _normalize(ref, dim=-1)

    # compute auto-correlation and cross-correlation
    # acf.shape = (..., 2 * filter_length, n_ref, n_est)
    # xcorr.shape = (..., filter_length, n_ref, n_est)
    acf, xcorr = compute_stats_2(ref, est, length=filter_length)

    diag_indices = list(range(ref.shape[-2]))

    if load_diag is not None:
        # the diagonal factor of the Toeplitz matrix is the first
        # coefficient of the acf
        acf[..., 0, diag_indices, diag_indices] += load_diag

    # solve for the SDR
    acf_sdr = acf[..., diag_indices, diag_indices]
    acf_sdr = acf_sdr.transpose(-2, -1)
    acf_sdr = acf_sdr[..., :filter_length]

    if pairwise:
        xcorr_sdr = xcorr.transpose(-3, -2)
    else:
        xcorr_sdr = xcorr[..., :, diag_indices, diag_indices].transpose(-2, -1)

    # solve for the optimal filter
    if use_cg_iter is None:
        # regular matrix solver
        R_mat = toeplitz(acf_sdr)
        sol = torch.linalg.solve(R_mat, xcorr_sdr)
    else:
        # use preconditioned conjugate gradient
        sol = toeplitz_conjugate_gradient(acf_sdr, xcorr_sdr, n_iter=use_cg_iter)

    # pairwise coherence
    if pairwise:
        coh_sdr = torch.einsum(
            "...lc,...lc->...c", xcorr_sdr, sol
        )  # (..., n_ref, n_est)
    else:
        coh_sdr = torch.einsum("...l,...l->...", xcorr_sdr, sol)

    # solve the coefficients for the SIR
    xcorr = xcorr.reshape(xcorr.shape[:-3] + (-1,) + xcorr.shape[-1:])
    if use_cg_iter is None:
        R_mat = block_toeplitz(acf)
        sol = torch.linalg.solve(R_mat, xcorr)
    else:
        if pairwise:
            x0 = sol
            x0 = sol.transpose(-3, -2)
            x0 = x0.reshape(x0.shape[:-3] + (-1,) + x0.shape[-1:])
        else:
            x0 = None
        sol = block_toeplitz_conjugate_gradient(acf, xcorr, n_iter=use_cg_iter, x=x0)
    coh_sar = torch.einsum("...lc,...lc->...c", xcorr, sol)

    if pairwise:
        coh_sdr, coh_sar = torch.broadcast_tensors(coh_sdr, coh_sar[..., None, :])

    return coh_sdr, coh_sar


def _base_metrics_bss_eval(
    coh_sdr: torch.Tensor, coh_sar: torch.Tensor, clamp_db: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function that converts the square cosine metrics to SDR/SIR/SAR
    """

    coh_sir = coh_sdr / coh_sar

    neg_sdr = _coherence_to_neg_sdr(coh_sdr, clamp_db=clamp_db)
    neg_sir = _coherence_to_neg_sdr(coh_sir, clamp_db=clamp_db)
    neg_sar = _coherence_to_neg_sdr(coh_sar, clamp_db=clamp_db)

    return neg_sdr, neg_sir, neg_sar


def bss_eval_sources(
    ref: torch.Tensor,
    est: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    compute_permutation: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes the SDR, SIR, and SAR for the input reference and estimated
    signals.

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    est: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
    filter_length: int, optional
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter: int, optional
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean: bool, optional
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db: bool, optional
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    compute_permutation: bool, optional
        When this flag is true, the optimal permutation of the estimated signals
        to maximize the SIR is computed (default: ``True``)
    load_diag: float, optional
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero

    Returns
    -------
    sdr: torch.Tensor, (..., n_channels_est)
        The signal-to-distortion-ratio (SDR)
    sir: torch.Tensor, (..., n_channels_est)
        The signal-to-interference-ratio (SIR)
    sar: torch.Tensor, (..., n_channels_est)
        The signal-to-interference-ratio (SIR)
    perm: torch.Tensor, (..., n_channels_est)
        The index of the corresponding reference signal (only when
        ``compute_permutation == True``)
    """

    if not compute_permutation:
        assert ref.shape[-2] == est.shape[-2], (
            "For non-pairwise computation, the number of reference"
            " and estimated signals must be the same"
        )

    coh_sdr, coh_sar = square_cosine_metrics(
        ref,
        est,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        pairwise=compute_permutation,
        load_diag=load_diag,
    )

    neg_sdr, neg_sir, neg_sar = _base_metrics_bss_eval(
        coh_sdr, coh_sar, clamp_db=clamp_db
    )

    if compute_permutation:
        neg_sir, neg_sdr, neg_sar, perm = _solve_permutation(
            neg_sir, neg_sdr, neg_sar, return_perm=True
        )
        return -neg_sdr, -neg_sir, -neg_sar, perm
    else:
        return -neg_sdr, -neg_sir, -neg_sar
