import pytest
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import fast_bss_eval
import fast_bss_eval.torch.hungarian as hungarian

import signal

class Timeout:
    """
    A class to catch timeout errors for the nan problem
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

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
def test_nan(seed, backend, n_chan=2, n_samples=4000):

    est = torch.zeros((n_chan, n_samples)).normal_()
    ref = torch.zeros((n_chan, n_samples)).normal_()

    # add one nan
    est[0, n_samples // 2] = torch.nan

    if backend == "numpy":
        est = est.numpy()
        ref = ref.numpy()

    with pytest.raises(ValueError):
        try:
            with Timeout(seconds=1):
                sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref, est)
        except TimeoutError:
            assert False, "Test nan: timeout exceeded"
