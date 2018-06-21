#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='uqef',
    version="0.2",
    url='https://gitlab.lrz.de/di73wal/UQEF',
    author="Florian Kuenzner",
    author_email="florian.kuenzner@tum.de",
    license='MIT',
    platforms='any',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "abc", "argparse", "chaospy", "dill", "enum", "glob", "itertools", "joblib", "json", "matplotlib", "more_itertools", "mpi4py", "multiprocessing", "numpy", "scipy", "setuptools", "tabulate", "time"
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
