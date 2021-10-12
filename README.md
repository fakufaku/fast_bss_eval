fast\_bss\_eval
==============

A package with a fast implementation of the bss\_eval metrics for the
evaluation of blind source separation.

Author
------

* [Robin Scheibler](robin.scheibler@linecorp.com)

Quick Start
-----------

### Install

```bash
# from pypi
pip install fast_bss_eval

# or from source
git clone https://github.com/fakufaku/fast_bss_eval
cd fast_bss_eval
pip install -e .
```

### Use

Assuming you have multichannel signals for the estmated and reference sources
stored in wav format files names ``my_estimate_file.wav`` and
``my_reference_file.wav``, respectively, you can quickly evaluate the bss_eval
metrics as follows.


```python
from scipy.io import wavfile
import fast_bss_eval

# open the files, we assume the sampling rate is known
# to be the same
fs, ref = wavfile.read("my_reference_file.wav")
_, est = wavfile.read("my_estimate_file.wav")

# compute the metrics
sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref.T, est.T)
```

Benchmark
---------

This package is significantly faster than other packages that also allow
to compute bss\_eval metrics such as [mir\_eval](https://github.com/craffel/mir_eval) or [sigsep/bsseva](https://github.com/sigsep/bsseval).
We did a benchmark using numpy/torch, single/double precision floating point
arithmetic (fp32/fp64, and using either Gaussian elimination or a [conjugate
gradient descent](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
(solve/CGD10).

<img src="./docs/figures/channels_vs_runtime.png">

License
-------

2021 (c) Robin Scheibler, LINE Corporation

This code is released under [MIT License](https://opensource.org/licenses/MIT).
