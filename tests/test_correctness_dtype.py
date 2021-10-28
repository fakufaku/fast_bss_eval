import pytest
import numpy as np
import torch
import mir_eval

import fast_bss_eval


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Was expecting a torch.Tensor or numpy.ndarray")


def max_error(values):
    values = [to_numpy(val) for val in values]
    errors = [np.max(np.abs(values[0] - val)) for val in values[1:]]
    return errors


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


def _as(to_fp32=False, to_torch=False, *args):
    if to_fp32:
        args = [arg.astype(np.float32) for arg in args]
    if to_torch:
        args = [torch.from_numpy(arg) for arg in args]

    if len(args) == 1:
        return args[0]
    else:
        return args


def compare_to_mir_eval(
    n_chan,
    n_samples,
    is_fp32,
    is_torch,
    tol,
    n_repeat,
    compute_permutation=True,
    verbose=False,
    **bss_eval_kwargs
):

    for n in range(n_repeat):

        _ref, _est = _random_input_pairs(1, n_chan, n_samples)

        # reference is mir_eval
        me_sdr, me_sir, me_sar, _ = mir_eval.separation.bss_eval_sources(
            _ref[0], _est[0], compute_permutation
        )

        ref, est = _as(is_fp32, is_torch, _ref, _est)

        sdr, sir, sar, *_ = fast_bss_eval.bss_eval_sources(
            ref, est, compute_permutation=compute_permutation, **bss_eval_kwargs
        )

        sdr = to_numpy(sdr)
        sir = to_numpy(sir)
        sar = to_numpy(sar)

        error_sdr = np.max(np.abs(sdr - me_sdr))
        error_sir = np.max(np.abs(sir - me_sir))
        error_sar = np.max(np.abs(sar - me_sar))

        if verbose:
            endofline = " " if compute_permutation else "\n"
            print(error_sdr, error_sir, error_sar, end=endofline)

        assert error_sdr < tol
        assert error_sir < tol
        assert error_sar < tol

        if compute_permutation == True:
            sdr_only = fast_bss_eval.sdr(ref, est, **bss_eval_kwargs)
            sdr_only = to_numpy(sdr_only)
            error_sdr_only = np.max(np.abs(sdr_only - me_sdr))
            assert error_sdr_only < tol
            if verbose:
                print(error_sdr_only)


@pytest.mark.parametrize(
    "is_fp32,is_torch,use_cg_iter,tol,compute_permutation",
    [
        (False, False, None, 1e-5, True),
        (True, False, None, 1e-2, True),
        (False, True, None, 1e-5, True),
        (True, True, None, 1e-2, True),
        (False, False, 20, 1e-1, True),
        (True, False, 20, 1e-1, True),
        (False, True, 20, 1e-1, True),
        (True, True, 20, 1e-1, True),
        (False, False, None, 1e-5, False),
        (False, True, None, 1e-5, False),
    ],
)
def test_accuracy(is_fp32, is_torch, use_cg_iter, tol, compute_permutation):
    """
    Tests that the output produced for random signals is
    reasonnably close to that produced by mir_eval

    Tests both ``bss_eval_sources`` and ``sdr``
    """
    np.random.seed(1)
    n = 1600
    c = 2
    n_repeat = 10
    verbose = True

    compare_to_mir_eval(
        c,
        n,
        is_fp32,
        is_torch,
        tol,
        n_repeat,
        verbose=verbose,
        compute_permutation=compute_permutation,
        use_cg_iter=use_cg_iter,
    )


@pytest.mark.parametrize(
    "is_torch,is_fp32,expected_dtype",
    [
        (False, False, np.float64),
        (True, False, torch.float64),
        (False, True, np.float32),
        (True, True, torch.float32),
    ],
)
def test_dtype(is_torch, is_fp32, expected_dtype):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 16000  # samples

    ref, est = _random_input_pairs(b, c, n, is_torch=is_torch, is_fp32=is_fp32)

    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est)
    assert sdr.dtype == expected_dtype
    assert sir.dtype == expected_dtype
    assert sar.dtype == expected_dtype


@pytest.mark.parametrize(
    "is_torch,is_fp32,clamp_db",
    [
        (False, False, 30),
        (True, False, 30),
        (False, True, 30),
        (True, True, 30),
    ],
)
def test_clamp_db(is_torch, is_fp32, clamp_db):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 16000  # samples

    ref, est = _random_input_pairs(
        b,
        c,
        n,
        is_torch=is_torch,
        is_fp32=is_fp32,
        matrix_factor=0.01,
        noise_factor=0.0,
    )
    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est, clamp_db=clamp_db)

    the_max = max([sdr.max(), sir.max(), sar.max()])
    the_min = max([sdr.min(), sir.min(), sar.min()])

    assert the_max <= clamp_db + 1
    assert the_min >= -clamp_db - 1


@pytest.mark.parametrize(
    "is_torch,is_fp32,clamp_db,use_cg_iter,zero_mean,load_diag",
    [
        (False, False, 30, None, False, None),
        (False, True, 30, None, False, None),
        (True, True, 30, None, False, None),
        (True, False, 30, None, False, None),
        (False, False, 30, None, False, 1e-5),
        (True, True, 30, None, False, 1e-5),
        (False, False, None, None, False, None),
        (True, True, None, None, False, None),
        (False, False, None, 10, False, None),
        (True, True, None, 10, False, None),
        (False, False, None, None, True, None),
        (True, True, None, None, True, None),
    ],
)
def test_sdr_loss(is_torch, is_fp32, clamp_db, use_cg_iter, zero_mean, load_diag):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 16000  # samples

    ref, est = _random_input_pairs(
        b,
        c,
        n,
        is_torch=is_torch,
        matrix_factor=0.1,
        noise_factor=0.01,
        is_fp32=is_fp32,
    )

    sdr1 = fast_bss_eval.sdr(
        ref,
        est,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    # sdr_loss does not solve for the permutation,
    # but the ordering of estimated sources matches
    # the references in _random_input_pairs
    sdr2 = fast_bss_eval.sdr_loss(
        est,
        ref,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    if isinstance(est, torch.Tensor):
        est_perm = est.flip(dims=(1,))
    else:
        est_perm = est[:, ::-1, :]
    sdr3 = fast_bss_eval.sdr_pit_loss(
        est_perm,
        ref,
        clamp_db=clamp_db,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
        load_diag=load_diag,
    )

    if is_fp32 or use_cg_iter is not None:
        tol = 1e-2
    else:
        tol = 1e-5

    assert abs(sdr1 + sdr2).max() < tol
    assert abs(sdr1 + sdr3).max() < tol


@pytest.mark.parametrize(
    "is_torch,is_fp32,clamp_db,zero_mean",
    [
        (False, False, 30, False),
        (False, True, 30, False),
        (True, True, 30, False),
        (True, False, 30, False),
        (False, False, None, False),
        (False, False, None, True),
        (True, True, None, False),
        (True, True, None, True),
    ],
)
def test_sdr_loss(is_torch, is_fp32, clamp_db, zero_mean):
    """
    Tests that the type of the output is the same as that of the input
    """
    b = 1  # batch
    c = 2  # channels
    n = 16000  # samples

    ref, est = _random_input_pairs(
        b,
        c,
        n,
        is_torch=is_torch,
        matrix_factor=0.1,
        noise_factor=0.01,
        is_fp32=is_fp32,
    )

    sdr1 = fast_bss_eval.si_sdr(
        ref,
        est,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    # sdr_loss does not solve for the permutation,
    # but the ordering of estimated sources matches
    # the references in _random_input_pairs
    sdr2 = fast_bss_eval.si_sdr_loss(
        est,
        ref,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    if isinstance(est, torch.Tensor):
        est_perm = est.flip(dims=(1,))
    else:
        est_perm = est[:, ::-1, :]

    sdr3 = fast_bss_eval.si_sdr_pit_loss(
        est_perm,
        ref,
        clamp_db=clamp_db,
        zero_mean=zero_mean,
    )

    sdr4, *_ = fast_bss_eval.si_bss_eval_sources(
        ref, est_perm, clamp_db=clamp_db, zero_mean=zero_mean
    )

    if is_fp32 is not None:
        tol = 1e-2
    else:
        tol = 1e-5

    assert abs(sdr1 + sdr2).max() < tol
    assert abs(sdr1 + sdr3).max() < tol
    assert abs(sdr1 - sdr4).max() < tol
