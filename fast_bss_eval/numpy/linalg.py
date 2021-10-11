import numpy as np


def hankel_view(x: np.ndarray, n_rows: int) -> np.ndarray:
    """return a view of x as a Hankel matrix"""
    n_cols = x.shape[-1] - n_rows + 1
    return np.lib.stride_tricks.as_strided(
        x, shape=x.shape[:-1] + (n_rows, n_cols), strides=x.strides + (x.strides[-1],)
    )


def toeplitz(x):
    """
    make symmetric toeplitz matrix
    """
    two_x = np.concatenate((x[..., :0:-1], x), axis=-1)
    H = hankel_view(two_x, n_rows=x.shape[-1])
    T = H[..., ::-1, :]
    return T


def block_toeplitz(x, n=None):
    """
    Create a block matrix where the blocks have a Toeplitz structure
    """
    top, rows, cols = x.shape[-3:]

    if n is None:
        n = top // 2

    out = np.zeros_like(x, shape=x.shape[:-3] + (n, rows, n, cols))

    for r in range(x.shape[-2]):
        for c in range(x.shape[-1]):
            vec = np.concatenate((x[..., -n + 1 :, r, c], x[..., :n, r, c]), axis=-1)
            T = hankel_view(vec, n)
            out[..., :, r, :, c] = T[..., ::-1, :]

    out = out.reshape(out.shape[:-4] + (n * rows, n * cols))

    return out
