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
    ref = np.random.randn(b, c, n)
    est = (
        np.eye(c) + matrix_factor * np.random.randn(b, c, c)
    ) @ ref + noise_factor * np.random.randn(b, c, n)

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


if __name__ == "__main__":

    n = 16000 * 5
    c = 2
    b = 1

    # reference is numpy 64 bits
    _ref, _est = _random_input_pairs(b, c, n)

    # reference is mir_eval
    me_sdr, me_sir, me_sar, _ = mir_eval.separation.bss_eval_sources(_ref[0], _est[0])

    sdrs = []
    sirs = []
    sars = []

    for is_fp32 in [False, True]:
        for is_torch in [False, True]:
            ref, est = _as(is_fp32, is_torch, _ref, _est)
            sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref, est)
            sdrs.append(sdr)
            sirs.append(sir)
            sars.append(sar)

    errors_sdr = max_error([me_sdr,] + sdrs)
    errors_sir = max_error([me_sir,] + sirs)
    errors_sar = max_error([me_sar,] + sars)

    print("Error SDR")
    print(errors_sdr)
    print()

    print("Error SIR")
    print(errors_sir)
    print()

    print("Error SAR")
    print(errors_sar)
    print()
