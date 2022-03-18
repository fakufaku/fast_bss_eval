import fast_bss_eval
import numpy as np
import pytest
import torch


def _random_input_pairs(
    nbatch,
    nchan,
    nsamples,
    is_fp32=False,
    is_torch=False,
    rand_perm=True,
):
    assert nsamples >= 2 * nchan

    # we create a lot of orthogonal signals
    signals = np.random.randn(nbatch, 2 * nchan, nsamples)

    cov = np.einsum("...cn,...dn->...cd", signals, signals) / nsamples
    ev, evec = np.linalg.eigh(cov)

    orth_signals = np.einsum(
        "...cd,...dn->...cn", evec.swapaxes(-2, -1) / np.sqrt(ev[..., None]), signals
    )

    cov_test = np.einsum("...cn, ...dn->...cd", orth_signals, orth_signals) / nsamples
    assert np.max(np.abs(cov_test - np.eye(2 * nchan))) < 1e-5

    ref = orth_signals[..., :nchan, :]
    noise = orth_signals[..., nchan:, :]

    # mixing coefficients
    # make sure the diagonal of alpha is larger than off-diag coeffecients
    alpha = np.random.rand(nbatch, nchan, nchan) - 0.5 + 2.0 * np.eye(nchan)
    alpha2 = alpha ** 2
    beta = np.random.randn(nbatch, nchan)
    beta2 = beta ** 2

    alpha2_diag = np.diagonal(alpha2, axis1=-2, axis2=-1)
    alpha2_sum = alpha2.sum(axis=-1)

    SDR = 10.0 * np.log10(alpha2_diag / (alpha2_sum - alpha2_diag + beta2))
    SIR = 10.0 * np.log10(alpha2_diag / (alpha2_sum - alpha2_diag))
    SAR = 10.0 * np.log10(alpha2_sum / beta2)

    est = np.einsum("...cd,...dn->...cn", alpha, ref) + beta[..., None] * noise

    # add a random permutation
    if rand_perm:
        perm = np.random.permutation(nchan)
    else:
        perm = np.arange(nchan)
    est = est[..., perm, :]

    inv_perm = np.empty(perm.size, dtype=perm.dtype)
    for i in np.arange(perm.size):
        inv_perm[perm[i]] = i
    inv_perm = inv_perm[None, :]

    # change type
    if is_fp32:
        est = est.astype(np.float32)
        ref = ref.astype(np.float32)
        SDR = SDR.astype(np.float32)
        SIR = SIR.astype(np.float32)
        SAR = SAR.astype(np.float32)

    if is_torch:
        est = torch.from_numpy(est)
        ref = torch.from_numpy(ref)
        SDR = torch.from_numpy(SDR)
        SIR = torch.from_numpy(SIR)
        SAR = torch.from_numpy(SAR)
        perm = torch.from_numpy(perm)

    return ref, est, SDR, SIR, SAR, inv_perm


@pytest.mark.parametrize(
    "is_torch,is_fp32,rand_perm,tol,nchan",
    [
        (True, True, True, 1e-1, 4),
        (True, True, False, 1e-1, 4),
        (True, False, False, 1e-5, 4),
        (True, False, True, 1e-5, 4),
        (False, False, True, 1e-5, 4),
        (False, True, True, 1e-1, 4),
        (False, True, False, 1e-1, 4),
        (False, False, False, 1e-5, 4),
    ],
)
def test_si_sdr(is_torch, is_fp32, rand_perm, tol, nchan):
    np.random.seed(10)

    nbatch = 10
    nsamples = 16000

    for trial in range(10):

        ref, est, SDR, SIR, SAR, PERM = _random_input_pairs(
            nbatch,
            nchan,
            nsamples,
            rand_perm=rand_perm,
            is_torch=is_torch,
            is_fp32=is_fp32,
        )

        sdr, sir, sar, *other = fast_bss_eval.si_bss_eval_sources(
            ref, est, compute_permutation=rand_perm
        )
        test_pairs = [
            ("sdr", sdr, SDR, tol),
            ("sir", sir, SIR, tol),
            ("sar", sar, SAR, 1e2 * tol),
        ]
        if rand_perm:
            test_pairs.append(("perm", other[0], PERM, 0))

        if rand_perm:
            sdr2, perm2 = fast_bss_eval.si_sdr(ref, est, return_perm=True)
            sdr3 = fast_bss_eval.si_sdr_pit_loss(est, ref)
            test_pairs += [
                ("sdr2", sdr2, SDR, tol),
                ("perm2", perm2, PERM, 0),
                ("sdr3", -sdr3, SDR, tol),
            ]
        else:
            sdr4 = fast_bss_eval.si_sdr_loss(est, ref)
            test_pairs += [("sdr4", -sdr4, SDR, tol)]

        for label, r1, r2, loc_tol in test_pairs:
            error = abs(r1 - r2).max()
            assert (
                error <= loc_tol
            ), f"{label} (error={error} > tol={loc_tol}) (trial={trial})"
