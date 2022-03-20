"""
Issue #5 on github reported some nan values in the output.

This seems to be an issue with using float32 in torch when the SDR
may be too high.
The metrics computation was modified to at least return an inf value
rather than nan.

This test verifies that no nan value is reported.
"""
import numpy as np
import torch
import pytest
from mir_eval.separation import bss_eval_sources
import fast_bss_eval

x = np.load("tests/data_issue_5.npz")
preds = torch.tensor(x["preds"])
target = torch.tensor(x["target"])

def to(x, backend, precision):

    if backend == "numpy":
        if precision == "single":
            return x.numpy().astype(np.float32)
        else:
            return x.numpy().astype(np.float64)

    elif backend.startswith("torch"):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if torch.cuda.is_available() and backend.endswith("cuda"):
            x = x.to(0)

        if precision == "single":
            return x.type(torch.float32)
        else:
            return x.type(torch.float64)

    else:
        raise ValueError("Invalid backend!")

@pytest.mark.parametrize(
    "backend, precision",
    [
        ("numpy", "single"),
        ("numpy", "double"),
        ("torch", "single"),
        ("torch", "double"),
        ("torch/cuda", "single"),
        ("torch/cuda", "double"),
    ],
)
def test_nan(backend, precision):

    if backend == "torch/cuda" and not torch.cuda.is_available():
        return

    tgt = to(target, backend, precision)
    prd = to(preds, backend, precision)

    outputs = fast_bss_eval.bss_eval_sources(tgt, prd)

    # the first 3 inputs are sdr, sir, sar
    # the 4th is perm, which we don't need to check
    for o in outputs[:3]:
        if backend.startswith("torch"):
            assert not torch.any(torch.isnan(o)) 
        else:
            assert not np.any(np.isnan(o)) 



if __name__ == "__main__":

    print(preds.shape, target.shape)
    print()

    for backend in ["numpy", "torch", "torch/cuda"]:

        if backend.endswith("cuda") and not torch.cuda.is_available():
            continue

        for precision in ["single", "double"]:
            tgt = to(target, backend, precision)
            prd = to(preds, backend, precision)

            print(backend, tgt.dtype)
            sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(tgt, prd)
            print(sdr, sdr.dtype)
            print()


    for precision in ["single", "double"]:
        tgt = to(target, "numpy", precision)
        prd = to(preds, "numpy", precision)

        print("mir_eval", tgt.dtype)
        sdr, _, _, _ = bss_eval_sources(tgt, prd, False)
        print(sdr, sdr.dtype)
        print()
