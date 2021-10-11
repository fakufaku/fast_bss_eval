import numpy as np
import torch
import fast_bss_eval

b = 1
c = 2
n = 16000

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


def test_dtype_numpy_64():
    ref, est = _random_input_pairs(b, c, n, is_torch=False, is_fp32=False)

    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est)
    assert sdr.dtype == np.float64
    assert sir.dtype == np.float64
    assert sar.dtype == np.float64

def test_dtype_torch_64():
    ref, est = _random_input_pairs(b, c, n, is_torch=True, is_fp32=False)

    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est)
    assert sdr.dtype == torch.float64
    assert sir.dtype == torch.float64
    assert sar.dtype == torch.float64

def test_dtype_numpy_32():
    ref, est = _random_input_pairs(b, c, n, is_torch=False, is_fp32=True)

    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est)
    assert sdr.dtype == np.float32
    assert sir.dtype == np.float32
    assert sar.dtype == np.float32

def test_dtype_torch_32():
    ref, est = _random_input_pairs(b, c, n, is_torch=True, is_fp32=True)

    sdr, sir, sar, _ = fast_bss_eval.bss_eval_sources(ref, est)
    assert sdr.dtype == torch.float32
    assert sir.dtype == torch.float32
    assert sar.dtype == torch.float32
