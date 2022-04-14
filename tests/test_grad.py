import fast_bss_eval
import mir_eval
import numpy as np
import pytest
import torch


def _zero_grad(*args):
    for arg in args:
        if hasattr(arg, "grad") and arg.grad is not None:
            arg.grad.zero_()


def _random_input_pairs(
    nbatch,
    nchan,
    nsamples,
    matrix_factor=0.5,
    noise_factor=0.1,
    is_fp32=False,
    is_torch=False,
):
    ref = np.random.randn(nbatch, nchan, nsamples)
    est = (
        np.eye(nchan) + matrix_factor * np.random.randn(nbatch, nchan, nchan)
    ) @ ref + noise_factor * np.random.randn(nbatch, nchan, nsamples)

    if is_fp32:
        est = est.astype(np.float32)
        ref = ref.astype(np.float32)

    if is_torch:
        est = torch.from_numpy(est)
        ref = torch.from_numpy(ref)

    return ref, est


@pytest.mark.parametrize(
    "is_fp32,clamp_db,use_cg_iter,zero_mean,load_diag",
    [
        (True, 30, None, False, None),
        (False, 30, None, False, None),
        (True, 30, None, False, 1e-5),
        (True, None, None, False, None),
        (True, None, 10, False, None),
        (True, None, None, True, None),
    ],
)
def test_sdr_loss(is_fp32, clamp_db, use_cg_iter, zero_mean, load_diag):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 8000  # samples

    ref, est = _random_input_pairs(
        b,
        c,
        n,
        is_torch=True,
        matrix_factor=0.1,
        noise_factor=0.01,
        is_fp32=is_fp32,
    )

    ref.requires_grad = True
    est.requires_grad = True

    """
    sdr, *_ = fast_bss_eval.bss_eval_sources(
        ref,
        est,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    loss = sdr.mean()
    loss.backward()

    _zero_grad(ref, est)
    """

    sdr = fast_bss_eval.sdr(
        ref,
        est,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    loss = sdr.mean()
    loss.backward()

    """
    _zero_grad(ref, est)

    # sdr_loss does not solve for the permutation,
    # but the ordering of estimated sources matches
    # the references in _random_input_pairs
    sdr = fast_bss_eval.sdr_loss(
        est,
        ref,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    loss = sdr.mean()
    loss.backward()

    _zero_grad(ref, est)

    if isinstance(est, torch.Tensor):
        est_perm = est.flip(dims=(1,))
    else:
        est_perm = est[:, ::-1, :]
    sdr = fast_bss_eval.sdr_pit_loss(
        est_perm,
        ref,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    loss = sdr.mean()
    loss.backward()
    """


@pytest.mark.parametrize(
    "is_fp32,clamp_db,zero_mean",
    [
        (True, 30, False),
        (False, 30, False),
        (True, None, False),
        (True, None, True),
    ],
)
def test_si_sdr_loss(is_fp32, clamp_db, zero_mean):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 8000  # samples

    ref, est = _random_input_pairs(
        b,
        c,
        n,
        is_torch=True,
        matrix_factor=0.1,
        noise_factor=0.01,
        is_fp32=is_fp32,
    )

    ref.requires_grad = True
    est.requires_grad = True

    sdr = fast_bss_eval.si_sdr(
        ref,
        est,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    loss = sdr.mean()
    loss.backward()

    _zero_grad(ref, est)

    # sdr_loss does not solve for the permutation,
    # but the ordering of estimated sources matches
    # the references in _random_input_pairs
    sdr = fast_bss_eval.si_sdr_loss(
        est,
        ref,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    loss = sdr.mean()
    loss.backward()

    _zero_grad(ref, est)

    if isinstance(est, torch.Tensor):
        est_perm = est.flip(dims=(1,))
    else:
        est_perm = est[:, ::-1, :]

    sdr = fast_bss_eval.si_sdr_pit_loss(
        est_perm,
        ref,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    loss = sdr.mean()
    loss.backward()

    _zero_grad(ref, est)

    sdr, *_ = fast_bss_eval.si_bss_eval_sources(
        ref, est_perm, clamp_db=clamp_db, zero_mean=zero_mean
    )

    loss = sdr.mean()
    loss.backward()


if __name__ == "__main__":
    test_sdr_loss(True, None, None, False, None)
