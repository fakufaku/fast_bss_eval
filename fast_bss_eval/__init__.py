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

r"""
This library implements the *bss_eval* metrics for the evaluation of blind
source separation (BSS) algorithms [1]_.  The implementation details and
benchmarks against other publicly available implementations are also available [2]_.

Quick start
-----------

The package can be installed from pip. At a minimum, it requires to have ``numpy``/``scipy`` installed.
To work with torch tensors, ``pytorch`` should also be installed obviously.

..  code-block:: shell

    pip install fast-bss-eval

Assuming you have multichannel signals for the estimated and reference sources
stored in wav format files named ``my_estimate_file.wav`` and
``my_reference_file.wav``, respectively, you can quickly evaluate the bss_eval
metrics as follows with the function :py:func:`bss_eval_sources` and :py:func:`sdr`.

.. code-block:: python

    from scipy.io import wavfile
    import fast_bss_eval

    # open the files, we assume the sampling rate is known
    # to be the same
    fs, ref = wavfile.read("my_reference_file.wav")
    _, est = wavfile.read("my_estimate_file.wav")

    # compute all the metrics
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref.T, est.T)

    # compute the SDR only
    sdr = fast_bss_eval.sdr(ref.T, est.T)

You can use the :py:func:`sdr_pit_loss` and :py:func:`sdr_loss` to train separation networks with
`pytorch`_ and
`torchaudio <https://github.com/pytorch/audio>`_.
The difference with the regular functions is as follows.

- reverse order of ``est`` and ``ref`` to follow pytorch convention
- sign of SDR is changed to act as a loss function

For permutation invariant training (PIT), use :py:func:`sdr_pit_loss`.  Use
:py:func:`sdr_loss` if the assignement of estimates to references is known, or
to compute the pairwise metrics.

.. code-block:: python

    import torchaudio

    # open the files, we assume the sampling rate is known
    # to be the same
    ref, fs = torchaudio.load("my_reference_file.wav")
    est, _ = torchaudio.load("my_estimate_file.wav")

    # PIT loss for permutation invariant training
    neg_sdr = fast_bss_eval.sdr_pit_loss(est, ref)

    # when the assignemnt of estimates to references is known
    neg_sdr = fast_bss_eval.sdr_loss(est, ref)

To summarize, there are four functions implemented, :func:`bss_eval_sources`,
:func:`sdr`, and :func:`sdr_pit_loss`, and :func:`sdr_loss`.
The **scale-invariant** [3]_ versions of all these functions are available in functions pre-fixed by ``si_``, i.e.,

* :py:func:`si_bss_eval_sources`
* :py:func:`si_sdr`
* :py:func:`si_sdr_pit_loss`
* :py:func:`si_sdr_loss`.

Alternatively, the regular functions with keyword argument ``filter_length=1`` can be used.

.. note::

    If you use this package in your own research, please cite `our paper <https://arxiv.org/abs/2110.06440>`_ describing it [4]_.

    .. code-block:: bibtex

        @misc{scheibler_sdr_2021,
          title={SDR --- Medium Rare with Fast Computations},
          author={Robin Scheibler},
          year={2021},
          eprint={2110.06440},
          archivePrefix={arXiv},
          primaryClass={eess.AS}
        }

Benchmark
---------

Average execution time on a dataset of multichannel signals with 2, 3, and 4
channels.  We compare ``fast_bss_eval`` to `mir_eval
<https://github.com/craffel/mir_eval>`_ and `sigsep
<https://github.com/sigsep/bsseval>`_.  We further benchmark ``fast_bss_eval``
using ``numpy``/``pytorch`` with double and single precision floating-point
arithmetic (fp64/fp32), and using ``solve`` function or an :ref:`iterative solver<Iterative solver for the disortion filters>` (CGD10).

The benchmark was run on a Linux workstation with an Intel Xeon Gold 6230 CPU @
2.10 GHz with 8 cores, an NVIDIA Tesla V100 GPU, and 64 GB of RAM.
We test single-threaded (1CPU) and 8 core performance (8CPU) on the CPU, and on the GPU.
We observe that for ``numpy`` there is not so much gain of using multiple cores.
It may be more efficient to run multiple samples in parallel single-threaded
processes to fully exploit multi-core architectures.


.. figure:: figures/channels_vs_runtime.png
    :alt: runtime benchmark result

bss_eval metrics theory
------------------------

:math:`\newcommand{\vs}{\boldsymbol{s}}\newcommand{\vshat}{\hat{\vs}}\newcommand{\vb}{\boldsymbol{b}}`
We consider the case where we have :math:`M` estimated signals :math:`\vshat_m`.
Each contains a convolutional mixture of :math:`K` reference signals
:math:`\vs_k` and an additive noise component :math:`\vb_m`,

.. math::

    \newcommand{\vh}{\boldsymbol{h}}
    \newcommand{\vg}{\boldsymbol{g}}

    \vshat_m = \sum\nolimits_{k}\vh_{mk} \star \vs_k + \vb_m,
    \quad m=1,\ldots,M,

where :math:`\vshat_m`, :math:`\vs_k`, and :math:`\vb_m` are all real vectors
of length :math:`T`.  The length of the impulse responses :math:`\vh_{mk}` is
assumed to be short compared to :math:`T`.  For simplicity, the convolution
operation here includes truncation to size :math:`T`.  In most cases, the
number of estimates and references is the same, i.e. :math:`M=K`.  We keep them
distinct for generality.

.. figure:: figures/bss_eval_metrics_projections.png
    :scale: 50 %
    :alt: figure describing how bss_eval metrics are derived

bss_eval decomposes the estimated signal into three *orthogonal* components corresponding to the
target, interference, and noise, as shown in the illustration above.

.. math::
    \newcommand{\ve}{\boldsymbol{e}}
    \newcommand{\mP}{\boldsymbol{P}}
    \newcommand{\mA}{\boldsymbol{A}}
    \newcommand{\starget}{\vs^{\text{target}}}
    \newcommand{\einterf}{\ve^{\text{interf}}}
    \newcommand{\eartif}{\ve^{\text{artif}}}

    \starget_{km} = \mP_{k} \vshat_m, \quad
    \einterf_{km} = \mP \vshat_m - \mP_k \vshat_m, \quad
    \eartif_{km} = \vshat_m - \mP \vshat_m.

Rather than projecting directly on the signals themselves, bss_eval allows for
a short distortion filter of length :math:`L` to be applied.
This corresponds to projection onto the subspace spanned by the :math:`L`
shifts of the signals.
Matrices :math:`\mP_k` and :math:`\mP` are orthogonal projection matrices onto
the subspace spaned by the :math:`L` shifts of :math:`\vs_k` and of all
references, respectively.  Let :math:`\mA_k \in \mathbb{R}^{(T + L - 1) \times
L}` contain the :math:`L` shifts of :math:`\vs_k` in its columns and :math:`\mA
= [\mA_1,\ldots,\mA_K]`, then

.. math::

    \mP_k = \mA_k(\mA_k^\top \mA_k)^{-1}\mA_k^\top,\quad
    \mP = \mA(\mA^\top \mA)^{-1}\mA^\top.

Then, SDR, SIR, and SAR, in decibels, are defined as follows,

.. math::

    \text{SDR}_{km} & = 10\log_{10} \frac{\| \starget_{km} \|^2}{\|\einterf_{km} + \eartif_{km} \|^2}, \\
    \text{SIR}_{km} & = 10\log_{10} \frac{\| \starget_{km} \|^2}{\| \einterf_{km} \|^2}, \\
    \text{SAR}_{km} & = 10\log_{10} \frac{\| \starget_{km} + \einterf_{km} \|^2}{\| \eartif_{m} \|^2}.

Finally, assuming for simplicity that :math:`K=M`, the permutation of the estimated
sources :math:`\pi\,:\,\{1,\ldots,K\} \to \{1,\ldots,K\}` that maximizes :math:`\sum_k \text{SIR}_{k\,\pi(k)}`
is chosen.
Once we have the matrix of SIR values, we can use the `linear_sum_assignement
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_
function from scipy to find this optimal permutation efficiently.

The scale-invariant SDR is equivalent to using a distortion filter length of 1
[3]_.  The use of longer filters has been shown to be beneficial when training
neural networks to assist beamforming algorithms [2]_ [4]_.

Iterative solver for the disortion filters
------------------------------------------

.. note::

    **tl;dr** If you need more speed, add the keyword argument ``use_cg_iter=10`` to any method call.

As described above, computing the bss_eval metrics requires to solve the systems

.. math::

    (\mA_k^\top \mA_k) \vh_{km} & = \mA_k^\top \vshat_m, \\
    (\mA^\top \mA) \vg_{m} & = \mA^\top \vshat_m, \\

for all :math:`k` and :math:`m`.  These system matrices are large and solving
directly, e.g., using ``numpy.linalg.solve``, can be expensive.  However, these
matrices are also structured (`Toeplitz
<https://en.wikipedia.org/wiki/Toeplitz_matrix>`_ and block-Toeplitz).  In this
case, we can apply a much more efficient iterative algorithm based on
`preconditioned conjugate gradient descent
<https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ [5]_.

The default behavior of ``fast_bss_eval`` is to use the direct solver (via the
``solve`` function of ``numpy`` or ``pytorch``) to ensure consistency with
the output of other packages.
However, if speed becomes an issue, it is possible to switch to the iterative
solver by adding the option ``use_cg_iter`` (USE Conjugate Gradient ITERations) to
any of the methods in ``fast_bss_eval``.
We found that using 10 iterations is a good trade-off between speed and accuracy,
i.e., ``use_cg_iter=10``.

Author
------

This package was written by `Robin Scheibler <http://www.robinscheibler.com>`_.

License
-------

2021 (c) Robin Scheibler, LINE Corporation

The source code for this package is released under `MIT License
<https://opensource.org/licenses/MIT>`_ and available on `github
<https://github.com/fakufaku/fast_bss_eval>`_.

Bibliography
------------

.. [1] E. Vincent, R. Gribonval, and C. Fevotte, *Performance measurement in
    blind audio source separation*, IEEE Trans. Audio Speech Lang. Process., vol.
    14, no. 4, pp. 1462–1469, Jun. 2006.

.. [2] R\. Scheibler, *SDR - Medium Rare with Fast Computations*, arXiv:2110.06440, 2021. `pdf <https://arxiv.org/abs/2110.06440>`_

.. [3] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, *SDR - Half-baked or Well Done?*,
    In Proc. IEEE ICASSP, Brighton, UK, May 2019.

.. [4] C. Boeddeker et al., *Convolutive transfer function invariant SDR training criteria
    for multi-channel reverberant speech separation*, in Proc. IEEE ICASSP, Toronto,
    CA, Jun. 2021, pp. 8428–8432.

.. [5] R. H. Chan and M. K. Ng, *Conjugate gradient methods for Toeplitz
    systems*, SIAM Review, vol. 38, no. 3, pp. 427–482, Sep. 1996.
"""
from typing import Optional, Union, Tuple

import numpy as np

try:
    import torch as pt

    has_torch = True
    from . import torch as torch
    from .torch import sdr_pit_loss, si_sdr_pit_loss
except ImportError:
    has_torch = False

    # dummy pytorch module
    class pt:
        class Tensor:
            def __init__(self):
                pass

    # dummy torch submodule
    class torch:
        bss_eval_sources = None
        sdr = None
        sdr_loss = None


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

    The order of ``ref``/``est`` follows the convention of bss_eval (i.e., ref first).

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
    permutation maximizing the sum of all SDRs. This is unlike :py:func:`bss_eval_sources`
    that uses the SIR.

    The order of ``ref``/``est`` follows the convention of bss_eval (i.e., ref first).

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
        If set to True, the sign is flipped and the negative SDR is returned
        (default: ``False``)

    Returns
    -------
    sdr:
        The SDR of the input signal ``shape == (..., n_channels_est)``
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
    Computes the negative signal-to-distortion ratio (SDR).

    This function is almost the same as :py:func:`fast_bss_eval.sdr` except for

    * negative sign to make it a *loss* function
    * ``est``/``ref`` arguments follow the convention of pytorch loss functions (i.e., ``est`` first).

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
        The negative SDR of the input signal. The returned tensor has

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


def sdr_pit_loss(
    est: Union[np.ndarray, pt.Tensor],
    ref: Union[np.ndarray, pt.Tensor],
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = None,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    load_diag: Optional[float] = None,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Computes the negative signal-to-distortion ratio (SDR) with the PIT loss
    (permutation invariant training).  This means the permutation of sources
    minimizing the loss is chosen.

    This function is similar to :py:func:`fast_bss_eval.sdr` and :py:func:`fast_bss_eval.sdr_loss`, but with,

    * negative sign to make it a *loss* function
    * ``est``/``ref`` arguments follow the convention of pytorch loss functions (i.e., ``est`` first).
    * pairwise and permutation solving is used (a.k.a. PIT, permutation invariant training)

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

    Returns
    -------
    :
        The negative SDR of the input signal after solving for the optimal permutation.
        This is equivalent to using ``sdr_loss`` with ``pairwise=True`` with a PIT loss.
        The returned tensor has

        * ``shape == (..., n_channels_est)`` if ``pairwise == False``
    """
    return _dispatch_backend(
        numpy.sdr_pit_loss,
        torch.sdr_pit_loss,
        est,
        ref,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        load_diag=load_diag,
    )


def si_bss_eval_sources(
    ref: Union[np.ndarray, pt.Tensor],
    est: Union[np.ndarray, pt.Tensor],
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    compute_permutation: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> Union[Tuple[np.ndarray, ...], Tuple[pt.Tensor, ...]]:
    """
    This function computes the scale-invariant metrics, SI-SDR, SI-SIR, and
    SI-SAR, for the input reference and estimated signals.

    The order of ``ref``/``est`` follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref:
        The estimated signals, ``shape == (..., n_channels_est, n_samples)``
    est:
        The groundtruth reference signals, ``shape ==  (..., n_channels_ref, n_samples)``
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
    si-sdr:
        The signal-to-distortion-ratio (SDR), ``shape == (..., n_channels_est)``
    si-sir:
        The signal-to-interference-ratio (SIR), ``shape == (..., n_channels_est)``
    si-sar:
        The signal-to-interference-ratio (SIR), ``shape == (..., n_channels_est)``
    perm:
        The index of the corresponding reference signal (only when
        ``compute_permutation == True``), ``shape == (..., n_channels_est)``
    """

    return _dispatch_backend(
        numpy.si_bss_eval_sources,
        torch.si_bss_eval_sources,
        ref,
        est,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        compute_permutation=compute_permutation,
        load_diag=load_diag,
    )


def si_sdr(
    ref: Union[np.ndarray, pt.Tensor],
    est: Union[np.ndarray, pt.Tensor],
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    return_perm: Optional[bool] = False,
    change_sign: Optional[bool] = False,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Compute the scale-invariant signal-to-distortion ratio (SI-SDR) only.

    This function computes the SDR for all pairs of est/ref signals and finds the
    permutation maximizing the sum of all SDRs. This is unlike :py:func:`fast_bss_eval.bss_eval_sources`
    that uses the SIR.

    The order of ``ref``/``est`` follows the convention of bss_eval (i.e., ref first).

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
    return_perm:
        If set to True, the optimal permutation of the estimated signals is
        also returned (default: ``False``)
    change_sign:
        If set to True, the sign is flipped and the negative SDR is returned
        (default: ``False``)

    Returns
    -------
    si-sdr:
        The SDR of the input signal ``shape == (..., n_channels_est)``
    perm:
        The index of the corresponding reference signal ``shape == (..., n_channels_est)``.
        Only returned if ``return_perm == True``
    """
    return _dispatch_backend(
        numpy.si_sdr,
        torch.si_sdr,
        ref,
        est,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        return_perm=return_perm,
        change_sign=change_sign,
    )


def si_sdr_loss(
    est: Union[np.ndarray, pt.Tensor],
    ref: Union[np.ndarray, pt.Tensor],
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    pairwise: Optional[bool] = False,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Computes the negative scale-invariant signal-to-distortion ratio (SI-SDR).

    This function is almost the same as :py:func:`fast_bss_eval.sdr` except for

    * negative sign to make it a *loss* function
    * ``est``/``ref`` arguments follow the convention of pytorch loss functions (i.e., ``est`` first).

    Parameters
    ----------
    est:
        The estimated signals ``shape == (..., n_channels_est, n_samples)``
    ref:
        The groundtruth reference signals ``shape == (..., n_channels_ref, n_samples)``
    zero_mean:
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db:
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``
    pairwise:
        If set to True, the metrics are computed for every est/ref signals pair (default: ``False``).
        When set to False, it is expected that ``n_channels_ref == n_channels_est``.

    Returns
    -------
    :
        The negative SI-SDR of the input signal. The returned tensor has

        * ``shape == (..., n_channels_est)`` if ``pairwise == False``
        * ``shape == (..., n_channels_ref, n_channels_est)`` if ``pairwise == True``.
    """
    return _dispatch_backend(
        numpy.si_sdr_loss,
        torch.si_sdr_loss,
        est,
        ref,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
    )


def si_sdr_pit_loss(
    est: Union[np.ndarray, pt.Tensor],
    ref: Union[np.ndarray, pt.Tensor],
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
) -> Union[np.ndarray, pt.Tensor]:
    """
    Computes the negative scale-invariant signal-to-distortion ratio (SDR) with
    the PIT loss (permutation invariant training).  This means the permutation
    of sources minimizing the loss is chosen.

    This function is similar to :py:func:`fast_bss_eval.sdr` and :py:func:`fast_bss_eval.sdr_loss`, but with,

    * negative sign to make it a *loss* function
    * ``est``/``ref`` arguments follow the convention of pytorch loss functions (i.e., ``est`` first).
    * pairwise and permutation solving is used (a.k.a. PIT, permutation invariant training)

    Parameters
    ----------
    est:
        The estimated signals ``shape == (..., n_channels_est, n_samples)``
    ref:
        The groundtruth reference signals ``shape == (..., n_channels_ref, n_samples)``
    zero_mean:
        When set to True, the mean of all signals is subtracted prior
        to computation of the metrics (default: ``False``)
    clamp_db:
        If provided, the resulting metrics are clamped to be in the range ``[-clamp_db, clamp_db]``

    Returns
    -------
    :
        The negative SDR of the input signal after solving for the optimal permutation.
        This is equivalent to using ``sdr_loss`` with ``pairwise=True`` with a PIT loss.
        The returned tensor has

        * ``shape == (..., n_channels_est)`` if ``pairwise == False``
    """
    return _dispatch_backend(
        numpy.si_sdr_pit_loss,
        torch.si_sdr_pit_loss,
        est,
        ref,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
    )
