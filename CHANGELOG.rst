Changelog
=========

All notable changes to `fast_bss_eval
<https://github.com/fakufaku/fast_bss_eval>`_ will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

`Unreleased`_
-------------

Nothing yet.

`0.1.2`_ - 2022-03-21
---------------------

Bugfix
~~~~~~

- Modified slightly metrics computation to avoid getting nan in some cases (issue #5)

`0.1.1`_ - 2022-03-18
---------------------

New
~~~

- Removed CI for python 3.6, Added for python 3.10

Bugfix
~~~~~~

- For pytorch, the permutation solver would go in an infinite loop when
  a nan value was present. Corrected and test added.
- Modified slightly metrics computation to avoid getting nan in some cases (issue #5)

`0.1.0`_ - 2021-10-28
---------------------

New
~~~

- Functions for scale-invariant metrics ``si_bss_eval_sources``, ``si_sdr``, ``si_sdr_loss``
- Functions for PIT loss ``sdr_pit_loss`` and ``si_sdr_pit_loss``

Bugfix
~~~~~~

- Removes a stray ``print`` statement from ``sdr_loss``


.. _Unreleased: https://github.com/fakufaku/fast_bss_eval/compare/v0.1.2...main
.. _0.1.2: https://github.com/fakufaku/fast_bss_eval/compare/v0.1.1...v0.1.2
.. _0.1.1: https://github.com/fakufaku/fast_bss_eval/compare/v0.1.0...v0.1.1
.. _0.1.0: https://github.com/fakufaku/fast_bss_eval/compare/v0.0.2...v0.1.0
