import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

try:
    long_description = (here / "README.md").read_text(encoding="utf-8")
except Exception:
    long_description = ""

setup(
    name="fast_bss_eval",
    version="0.1.1",
    description="Package for fast computation of BSS Eval metrics for source separation",
    long_description=long_description,
    long_description_content_type="text/markdown",  # text/plain, text/x-rst, text/markdown
    url="https://github.com/fakufaku/fast_bss_eval",
    author="Robin Scheibler",
    author_email="fakufaku@gmail.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # keywords='sample, setuptools, development',
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    python_requires=">=3.6, <4",
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "all": [],
    },
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # data_files=[('my_data', ['data/data_file'])],
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/fakufaku/fast_bss_eval/issues",
        "Source": "https://github.com/fakufaku/fast_bss_eval/",
    },
)
