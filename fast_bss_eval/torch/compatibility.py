# Copyright 2022 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This file implements a number of wrappers for pytorch functions
whose API differs in older versions.
- rfft
- irfft
- solve
- division of complex by real
"""
try:
    from packaging.version import Version
except [ImportError, ModuleNotFoundError]:
    from distutils.version import LooseVersion as Version

import torch

is_torch_1_7_plus = Version(torch.__version__) >= Version("1.7.0")
is_torch_1_8_plus = Version(torch.__version__) >= Version("1.8.0")

# We define wrappers for rfft /irfft for older version of torch
if not is_torch_1_7_plus:

    def rfft(x, n=None, dim=-1):

        x = x.transpose(dim, -1)

        if n is not None:
            if n > x.shape[-1]:
                x = torch.nn.functional.pad(x, (0, n - x.shape[-1]))
            elif n < x.shape[-1]:
                x = x[..., :n]

        x = torch.rfft(x, 1)
        x = torch.view_as_complex(x)

        x = x.transpose(dim, -1)

        return x

    def irfft(x, n=None, dim=-1):
        x = x.transpose(dim, -1)

        if x.dtype not in [torch.complex64, torch.complex128]:
            x = 1j * x

        if n is not None:
            signal_sizes = torch.Size([n])
            n_freq = n // 2 + 1
            if n_freq < x.shape[-1]:
                x = x[..., :n_freq]
            elif n_freq > x.shape[-1]:
                x = torch.nn.functional.pad(x, (0, n_freq - x.shape[-1]))

        else:
            signal_sizes = torch.Size([(x.shape[-1] - 1) * 2])

        x = torch.irfft(torch.view_as_real(x), 1, signal_sizes=signal_sizes)

        x = x.transpose(dim, -1)
        return x

else:
    import torch.fft

    def rfft(*args, **kwargs):
        return torch.fft.rfft(*args, **kwargs)

    def irfft(*args, **kwargs):
        return torch.fft.irfft(*args, **kwargs)


# We define wrappers for solve and divide for older version of torch
if not is_torch_1_8_plus:

    def solve(A, b):
        pad_b = b.ndim == A.ndim - 1
        if pad_b:
            b = b[..., None]

        x, _ = torch.solve(b, A)

        if pad_b:
            x = x[..., 0]

        return x

    def inv(A):
        return torch.inverse(A)

    def divide_complex_real(val_complex, val_real):
        val_complex = torch.view_as_real(val_complex)
        return torch.view_as_complex(val_complex / val_real[..., None])

else:

    def solve(*args, **kwargs):
        return torch.linalg.solve(*args, **kwargs)

    def inv(*args, **kwargs):
        return torch.linalg.inv(*args, **kwargs)

    def divide_complex_real(val_complex, val_real):
        return val_complex / val_real


def einsum_complex(expr, op1, op2):

    rr = torch.einsum(expr, op1.real, op2.real)
    ri = torch.einsum(expr, op1.real, op2.imag)
    ir = torch.einsum(expr, op1.imag, op2.real)
    ii = torch.einsum(expr, op1.imag, op2.imag)

    res = torch.stack([rr - ii, ri + ir], dim=-1)
    res = torch.view_as_complex(res)

    return res
