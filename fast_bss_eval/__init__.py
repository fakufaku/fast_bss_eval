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

"""
This library implements the *bss_eval* metrics for the evaluation of blind
source separation (BSS) algorithms [1]_.  The implementation details and
benchmarks against other publicly available implementations are also available [2]_.

Quick start
-----------

The package can be installed from pip. At a minimum, it requires to have ``numpy``/``scipy`` installed.
To work with torch tensors, ``pytorch`` should also be installed obviously.

..  code-block:: shell

    pip install fast_bss_eval

Assuming you have multichannel signals for the estmated and reference sources
stored in wav format files names ``my_estimate_file.wav`` and
``my_reference_file.wav``, respectively, you can quickly evaluate the bss_eval
metrics as follows.

.. code-block:: python

    from scipy.io import wavfile
    import fast_bss_eval

    # open the files, we assume the sampling rate is known
    # to be the same
    fs, ref = wavfile.read("my_reference_file.wav")
    _, est = wavfile.read("my_estimate_file.wav")

    # compute the metrics
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref.T, est.T)

There are three functions implemented, :func:`fast_bss_eval.bss_eval_sources`,
:func:`fast_bss_eval.sdr`, and :func:`fast_bss_eval.sdr_loss`.

Benchmark
---------

Average execution time on a dataset of multichannel signals with 2, 3, and 4
channels.

.. figure:: figures/channels_vs_runtime.png
    :alt: runtime benchmark result

bss_eval metrics theory
------------------------

Basics
~~~~~~

TBA.

.. figure:: figures/bss_eval_metrics_projections.png
    :scale: 50 %
    :alt: figure describing how bss_eval metrics are derived

The scale-invariant SDR is equivalent to using a distortion filter length of 1
[3]_.  The use of longer filters has been shown to be beneficial when training
neural networks to assist beamforming algorithms [2]_ [4]_.

Iterative solver for the disortion filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``use_cg_iter=10`` option when using any of the functions.

References
----------

.. [1] E. Vincent, R. Gribonval, and C. Fevotte, *Performance measurement in
    blind audio source separation*, IEEE Trans. Audio Speech Lang. Process., vol.
    14, no. 4, pp. 1462–1469, Jun. 2006.

.. [2] R. Scheibler, *SDR &mdash; Medium Rare with Fast Computations*, arXiv, 2021.

.. [3] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, *SDR &mdash; Half-baked or Well Done?*,
    In Proc. IEEE ICASSP, Brighton, UK, May 2019.

.. [4] C. Boeddeker et al., *Convolutive transfer function invariant SDR training criteria
    for multi-channel reverberant speech separation*, in Proc. IEEE ICASSP, Toronto,
    CA, Jun. 2021, pp. 8428–8432.
"""
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
