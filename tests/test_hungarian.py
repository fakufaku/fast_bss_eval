import pytest
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
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
