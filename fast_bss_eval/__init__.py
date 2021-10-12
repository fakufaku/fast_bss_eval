from typing import Optional, Union, Tuple

import numpy as np

try:
    import torch as pt

    has_torch = True
    from . import torch as torch
except ImportError:
    has_torch = False

    class pt:
        class Tensor:
            def __init__(self):
                pass


from . import numpy as numpy


def _dispatch_backend(f_numpy, f_torch, *args, **kwargs):
    if has_torch and isinstance(args[0], pt.Tensor):
        return f_torch(*args, **kwargs)
    else:
        return f_numpy(*args, **kwargs)


def bss_eval_sources(
    ref: Union[np.ndarray, pt.Tensor],
    est: Union[np.ndarray, pt.Tensor],
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    compute_permutation: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> Union[Tuple[np.ndarray, ...], Tuple[pt.Tensor, ...]]:
    """
    This function computes the SDR, SIR, and SAR for the input reference and estimated
    signals.

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref:
        The estimated signals, ``shape == (..., n_channels_est, n_samples)``
    est:
        The groundtruth reference signals, ``shape ==  (..., n_channels_ref, n_samples)``
    filter_length:
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter:
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean:
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db:
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    compute_permutation:
        When this flag is true, the optimal permutation of the estimated signals
        to maximize the SIR is computed (default: ``True``)
    load_diag:
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero

    Returns
    -------
    sdr:
        The signal-to-distortion-ratio (SDR), ``shape == (..., n_channels_est)``
    sir:
        The signal-to-interference-ratio (SIR), ``shape == (..., n_channels_est)``
    sar:
        The signal-to-interference-ratio (SIR), ``shape == (..., n_channels_est)``
    perm:
        The index of the corresponding reference signal (only when
        ``compute_permutation == True``), ``shape == (..., n_channels_est)``
    """

    return _dispatch_backend(
        numpy.bss_eval_sources,
        torch.bss_eval_sources,
        ref,
        est,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        compute_permutation=compute_permutation,
        load_diag=load_diag,
    )


def sdr(
    ref: Union[np.ndarray, pt.Tensor],
    est: Union[np.ndarray, pt.Tensor],
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
    return_perm: Optional[bool] = False,
    change_sign: Optional[bool] = False,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Compute the signal-to-distortion ratio (SDR) only.

    This function computes the SDR for all pairs of est/ref signals and finds the
    permutation maximizing the sum of all SDRs.

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref:
        The estimated signals, ``shape == (..., n_channels_est, n_samples)``
    est:
        The groundtruth reference signals, ``shape == (..., n_channels_ref, n_samples)``
    filter_length:
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter:
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean:
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db:
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    load_diag:
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    return_perm:
        If set to True, the optimal permutation of the estimated signals is
        also returned (default: ``False``)
    change_sign:
        If set to True, the sign is flipped and the negative CI-SDR is returned
        (default: ``False``)

    Returns
    -------
    cisdr:
        The CI-SDR of the input signal ``shape == (..., n_channels_est)``
    perm:
        The index of the corresponding reference signal ``shape == (..., n_channels_est)``.
        Only returned if ``return_perm == True``
    """
    return _dispatch_backend(
        numpy.sdr,
        torch.sdr,
        ref,
        est,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        load_diag=load_diag,
        return_perm=return_perm,
        change_sign=change_sign,
    )


def sdr_loss(
    est: Union[np.ndarray, pt.Tensor],
    ref: Union[np.ndarray, pt.Tensor],
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
    pairwise: Optional[bool] = False,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Compute the negative signal-to-distortion ratio (CI-SDR).

    The order of est/ref follows the convention of pytorch loss functions
    (i.e., est first).

    Parameters
    ----------
    est:
        The estimated signals ``shape == (..., n_channels_est, n_samples)``
    ref:
        The groundtruth reference signals ``shape == (..., n_channels_ref, n_samples)``
    filter_length:
        The length of the distortion filter allowed (default: ``512``)
    use_cg_iter:
        If provided, an iterative method is used to solve for the distortion
        filter coefficients instead of direct Gaussian elimination.
        This can speed up the computation of the metrics in case the filters
        are long. Using a value of 10 here has been shown to provide
        good accuracy in most cases and is sufficient when using this
        loss to train neural separation networks.
    zero_mean:
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db:
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    load_diag:
        If provided, this small value is added to the diagonal coefficients of
        the system metrics when solving for the filter coefficients.
        This can help stabilize the metric in the case where some of the reference
        signals may sometimes be zero
    pairwise:
        If set to True, the metrics are computed for every est/ref signals pair (default: ``False``).
        When set to False, it is expected that ``n_channels_ref == n_channels_est``.

    Returns
    -------
    :
        The negative CI-SDR of the input signal. The returned tensor has

        * ``shape == (..., n_channels_est)`` if ``pairwise == False``
        * ``shape == (..., n_channels_ref, n_channels_est)`` if ``pairwise == True``.
    """
    return _dispatch_backend(
        numpy.sdr_loss,
        torch.sdr_loss,
        est,
        ref,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
        load_diag=load_diag,
    )
