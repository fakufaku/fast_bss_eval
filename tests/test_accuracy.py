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

        sdr = to_numpy(sdr)
        sir = to_numpy(sir)
        sar = to_numpy(sar)

        error_sdr = np.max(np.abs(sdr - me_sdr))
        error_sir = np.max(np.abs(sir - me_sir))
        error_sar = np.max(np.abs(sar - me_sar))

        if verbose:
            print(error_sdr, error_sir, error_sar)

        assert error_sdr < tol
        assert error_sir < tol
        assert error_sar < tol


def test_accuracy():
    np.random.seed(1)
    n = 1600
    c = 2
    tol = 1e-1
    n_repeat = 10
    use_cg_iter = 30
    verbose = True

    # numpy / fp32
    compare_to_mir_eval(c, n, False, False, tol, n_repeat, verbose=verbose)
    # numpy / fp64
    compare_to_mir_eval(c, n, True, False, tol, n_repeat, verbose=verbose)
    # numpy / fp32
    compare_to_mir_eval(c, n, False, True, tol, n_repeat, verbose=verbose)
    # numpy / fp32
    compare_to_mir_eval(c, n, True, True, tol, n_repeat, verbose=verbose)

    # numpy / fp32
    compare_to_mir_eval(
        c, n, False, False, tol, n_repeat, verbose=verbose, use_cg_iter=use_cg_iter
    )
    # numpy / fp64
    compare_to_mir_eval(
        c, n, True, False, tol, n_repeat, verbose=verbose, use_cg_iter=use_cg_iter
    )
    # numpy / fp32
    compare_to_mir_eval(
        c, n, False, True, tol, n_repeat, verbose=verbose, use_cg_iter=use_cg_iter
    )
    # numpy / fp32
    compare_to_mir_eval(
        c, n, True, True, tol, n_repeat, verbose=verbose, use_cg_iter=use_cg_iter
    )


if __name__ == "__main__":
    test_accuracy()
