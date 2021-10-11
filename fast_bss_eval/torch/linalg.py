import torch


def hankel_view(x: torch.Tensor, n_rows: int) -> torch.Tensor:
    """return a view of x as a Hankel matrix"""
    n_cols = x.shape[-1] - n_rows + 1
    x_strides = x.stride()
    return torch.as_strided(
        x, size=x.shape[:-1] + (n_rows, n_cols), stride=x_strides + (x_strides[-1],)
    )


def toeplitz(x):
    """
    make symmetric toeplitz matrix
    """
    two_x = torch.cat((x[..., 1:].flip(dims=(-1,)), x), dim=-1)
    H = hankel_view(two_x, n_rows=x.shape[-1])
    T = H.flip(dims=(-2,))
    return T


def block_toeplitz(x, n=None):
    """
    Create a block matrix where the blocks have a Toeplitz structure
    """
    top, rows, cols = x.shape[-3:]

    if n is None:
        n = top // 2

    out = x.new_zeros(x.shape[:-3] + (n, rows, n, cols))

    for r in range(x.shape[-2]):
        for c in range(x.shape[-1]):
            vec = torch.cat((x[..., -n + 1 :, r, c], x[..., :n, r, c]), dim=-1)
            T = hankel_view(vec, n).flip(dims=(-2,))
            out[..., :, r, :, c] = T

    out = out.reshape(out.shape[:-4] + (n * rows, n * cols))

    return out
