import fast_bss_eval
import fast_bss_eval.torch.hungarian as hungarian
import numpy as np
import pytest
import torch
from fast_bss_eval.torch.helpers import _linear_sum_assignment_with_inf
from scipy.optimize import linear_sum_assignment


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize("cols", [2, 3, 4, 5, 6, 10])
def test_hungarian(seed, rows, cols):

    np.random.seed(seed)
    C = np.random.randn(rows, cols)

    row1, col1 = hungarian.linear_sum_assignment(torch.from_numpy(C))
    row2, col2 = linear_sum_assignment(C)

    assert np.all(row2 - row1.cpu().numpy() == 0) and np.all(
        col2 - col1.cpu().numpy() == 0
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.timeout(2)
def test_nan(seed, backend, n_chan=2, n_samples=4000):

    est = torch.zeros((n_chan, n_samples)).normal_()
    ref = torch.zeros((n_chan, n_samples)).normal_()

    # add one nan
    # float("nan") is more backward compatible than torch.nan
    est[0, n_samples // 2] = float("nan")

    if backend == "numpy":
        est = est.numpy()
        ref = ref.numpy()

    with pytest.raises(ValueError):
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref, est)


def test_empty_input():
    cost_matrix = torch.zeros(0)
    out1, out2 = _linear_sum_assignment_with_inf(cost_matrix)

    assert out1.numel() == 0
    assert out2.numel() == 0
