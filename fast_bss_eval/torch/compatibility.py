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
- einsum
For torch<=1.5 the torch_complex library is used.
"""
try:
    from packaging.version import Version
except [ImportError, ModuleNotFoundError]:
    from distutils.version import LooseVersion as Version

from torch_complex import ComplexTensor

import torch

is_torch_1_8_plus = Version(torch.__version__) >= Version("1.8.0")

if not is_torch_1_8_plus:
    try:
        import torch_complex
    except ImportError:
        raise ImportError(
            "When using torch<=1.7, the package torch_complex is required."
            " Install it as `pip install torch_complex`"
        )

# We define wrappers for rfft /irfft for older version of torch
if not is_torch_1_8_plus:

    def rfft(x, n=None, dim=-1):

        x = x.transpose(dim, -1)

        if n is not None:
            if n > x.shape[-1]:
                x = torch.nn.functional.pad(x, (0, n - x.shape[-1]))
            elif n < x.shape[-1]:
                x = x[..., :n]

        x = torch.rfft(x, 1)

        x = torch_complex.ComplexTensor(x[..., 0], x[..., 1])

        x = x.transpose(dim, -1)

        return x

    def irfft(x, n=None, dim=-1):
        x = x.transpose(dim, -1)

        if isinstance(x, torch_complex.ComplexTensor):
            x = torch.stack((x.real, x.imag), dim=-1)
        else:
            x = torch.stack((x, x.new_zeros(x.shape)), dim=-1)

        # x is now in the old pytorch complex format with the last
        # dimension of size two containing real/imag separately

        if n is not None:
            signal_sizes = torch.Size([n])
            n_freq = n // 2 + 1
            if n_freq < x.shape[-2]:
                x = x[..., :n_freq, :]
            elif n_freq > x.shape[-2]:
                x = torch.nn.functional.pad(x, (0, n_freq - x.shape[-2], 0, 0))

        else:
            signal_sizes = torch.Size([(x.shape[-1] - 1) * 2])

        x = torch.irfft(x, 1, signal_sizes=signal_sizes)

        # x is now a real tensor here

        x = x.transpose(dim, -1)
        return x

else:
    import torch.fft

    def rfft(*args, **kwargs):
        return torch.fft.rfft(*args, **kwargs)

    def irfft(x, *args, **kwargs):
        # we add the check here because in torch>=1.7, if the input to
        # irfft is a real tensor, a spurious warning is generated
        if x.dtype not in [torch.complex64, torch.complex128]:
            x = x + 0.0j
        return torch.fft.irfft(x, *args, **kwargs)


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
        if isinstance(A, torch_complex.ComplexTensor):
            return A.inverse()
        else:
            return torch.inverse(A)

    def einsum(expr, *operands):
        if isinstance(operands[0], torch_complex.ComplexTensor):
            return torch_complex.einsum(expr, *operands)
        else:
            return torch.einsum(expr, *operands)

else:

    def solve(*args, **kwargs):
        return torch.linalg.solve(*args, **kwargs)

    def inv(*args, **kwargs):
        return torch.linalg.inv(*args, **kwargs)

    def einsum(*args, **kwargs):
        return torch.einsum(*args, **kwargs)


def einsum_complex(expr, op1, op2):

    rr = torch.einsum(expr, op1.real, op2.real)
    ri = torch.einsum(expr, op1.real, op2.imag)
    ir = torch.einsum(expr, op1.imag, op2.real)
    ii = torch.einsum(expr, op1.imag, op2.imag)

    res = torch.stack([rr - ii, ri + ir], dim=-1)
    res = torch.view_as_complex(res)

    return res
