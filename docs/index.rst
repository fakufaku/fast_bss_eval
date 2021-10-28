.. fast_bss_eval documentation master file, created by
   sphinx-quickstart on Mon Oct  4 09:37:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fast_bss_eval's documentation!
=========================================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   changelog

.. image:: https://readthedocs.org/projects/fast-bss-eval/badge/?version=latest
   :target: https://fast-bss-eval.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://github.com/fakufaku/fast_bss_eval/actions/workflows/lint.yml/badge.svg?branch=main
   :target: https://github.com/fakufaku/fast_bss_eval/actions/workflows/lint.yml
   :alt: Linting Status
.. image:: https://github.com/fakufaku/fast_bss_eval/actions/workflows/pythonpackage.yml/badge.svg
   :target: https://github.com/fakufaku/fast_bss_eval/actions/workflows/pythonpackage.yml
   :alt: Tests Status

| Do you have a zillion BSS audio files to process and it is taking days ?
| Is your simulation never ending ?
| 
| Fear no more! `fast_bss_eval` is here to help **you!**

``fast_bss_eval`` is a fast implementation of the bss_eval metrics for the
evaluation of blind source separation.  Our implementation of the bss\_eval
metrics has the following advantages compared to other existing ones.

* seamlessly works with **both** `numpy <https://numpy.org/>`_ arrays and `pytorch <https://pytorch.org>`_ tensors
* very fast
* can be even faster by using an iterative solver (add ``use_cg_iter=10`` option to the function call)
* supports batched computations
* differentiable via pytorch
* can run on GPU via pytorch

.. automodule:: fast_bss_eval

API
~~~

.. autofunction:: fast_bss_eval.bss_eval_sources

.. autofunction:: fast_bss_eval.sdr

.. autofunction:: fast_bss_eval.sdr_pit_loss

.. autofunction:: fast_bss_eval.sdr_loss

.. autofunction:: fast_bss_eval.si_bss_eval_sources

.. autofunction:: fast_bss_eval.si_sdr

.. autofunction:: fast_bss_eval.si_sdr_pit_loss

.. autofunction:: fast_bss_eval.si_sdr_loss

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
