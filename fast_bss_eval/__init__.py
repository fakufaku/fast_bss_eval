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
        The signal-to-distortion-ratio (SDR), ``shape == , (..., n_channels_est)``
    sir:
        The signal-to-interference-ratio (SIR), ``shape == , (..., n_channels_est)``
    sar:
        The signal-to-interference-ratio (SIR), ``shape == , (..., n_channels_est)``
    perm:
        The index of the corresponding reference signal (only when
        ``compute_permutation == True``), ``shape == , (..., n_channels_est)``
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
