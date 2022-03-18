import pytest
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import fast_bss_eval
import fast_bss_eval.torch.hungarian as hungarian


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
    est[0, n_samples // 2] = torch.nan

    if backend == "numpy":
        est = est.numpy()
        ref = ref.numpy()

    with pytest.raises(ValueError):
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref, est)
