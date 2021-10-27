from typing import Optional, Tuple
import torch

from .metrics import sdr_loss, sdr, bss_eval_sources


def si_sdr_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    pairwise: Optional[bool] = False,
) -> torch.Tensor:
    return sdr_loss(
        est,
        ref,
        filter_length=1,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
    )


def si_sdr_pit_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
) -> torch.Tensor:
    return sdr(
        ref,
        est,
        filter_length=1,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        return_perm=False,
        change_sign=True,
    )


def pairwise_si_sdr_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
) -> torch.Tensor:
    return si_sdr_loss(est, ref, zero_mean=zero_mean, clamp_db=clamp_db, pairwise=True)


def si_sdr(
    ref: torch.Tensor,
    est: torch.Tensor,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    return_perm: Optional[bool] = False,
    change_sign: Optional[bool] = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute the scale-invariant signal-to-distortion ratio (SI-SDR) only.

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
    return sdr(
        ref,
        est,
        filter_length=1,
        use_cg_iter=None,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        return_perm=return_perm,
        change_sign=change_sign,
    )


def si_bss_eval_sources(
    ref: torch.Tensor,
    est: torch.Tensor,
    zero_mean: Optional[bool] = False,
    clamp_db: Optional[float] = None,
    compute_permutation: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes the SI-SDR, SI-SIR, and SI-SAR for the input
    reference and estimated signals.

    The order of ref/est follows the convention of bss_eval (i.e., ref first).

    Parameters
    ----------
    ref: torch.Tensor (..., n_channels_est, n_samples)
        The estimated signals
    est: torch.Tensor (..., n_channels_ref, n_samples)
        The groundtruth reference signals
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
    si-sdr: torch.Tensor, (..., n_channels_est)
        The scale-invariant signal-to-distortion-ratio (SDR)
    si-sir: torch.Tensor, (..., n_channels_est)
        The scale-invariant signal-to-interference-ratio (SIR)
    si-sar: torch.Tensor, (..., n_channels_est)
        The scale-invariant signal-to-interference-ratio (SIR)
    perm: torch.Tensor, (..., n_channels_est)
        The index of the corresponding reference signal (only when
        ``compute_permutation == True``)
    """

    return bss_eval_sources(
        ref,
        est,
        filter_length=1,
        use_cg_iter=None,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        compute_permutation=compute_permutation,
        load_diag=load_diag,
    )
