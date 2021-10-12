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
    verbose=False,
    **bss_eval_kwargs
):

    for n in range(n_repeat):

        _ref, _est = _random_input_pairs(1, n_chan, n_samples)

        # reference is mir_eval
        me_sdr, me_sir, me_sar, _ = mir_eval.separation.bss_eval_sources(
            _ref[0], _est[0]
        )

        ref, est = _as(is_fp32, is_torch, _ref, _est)

        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
            ref, est, **bss_eval_kwargs
        )

        sdr_only = fast_bss_eval.sdr(ref, est, **bss_eval_kwargs)

        sdr = to_numpy(sdr)
        sir = to_numpy(sir)
        sar = to_numpy(sar)
        sdr_only = to_numpy(sdr_only)

        error_sdr = np.max(np.abs(sdr - me_sdr))
        error_sir = np.max(np.abs(sir - me_sir))
        error_sar = np.max(np.abs(sar - me_sar))
        error_sdr_only = np.max(np.abs(sdr_only - me_sdr))

        if verbose:
            print(error_sdr, error_sir, error_sar, error_sdr_only)

        assert error_sdr < tol
        assert error_sir < tol
        assert error_sar < tol
        assert error_sdr_only < tol


@pytest.mark.parametrize(
    "is_fp32,is_torch,use_cg_iter,tol",
    [
        (False, False, None, 1e-5),
        (True, False, None, 1e-2),
        (False, True, None, 1e-5),
        (True, True, None, 1e-2),
        (False, False, 20, 1e-1),
        (True, False, 20, 1e-1),
        (False, True, 20, 1e-1),
        (True, True, 20, 1e-1),
    ],
)
def test_accuracy(is_fp32, is_torch, use_cg_iter, tol):
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
        c, n, is_fp32, is_torch, tol, n_repeat, verbose=verbose, use_cg_iter=use_cg_iter
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
