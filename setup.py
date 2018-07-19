#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='uqef',
    version="0.2",
    url='https://gitlab.lrz.de/di73wal/UQEF',
    author="Florian Kuenzner",
    author_email='florian.kuenzner@tum.de',
    license='MIT',
    platforms='any',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        'argparse', 'chaospy', 'dill', 'joblib', 'matplotlib', 'more_itertools', 'mpi4py', 'numpy', 'scipy', 'setuptools', 'tabulate',
        'enum;python_version<"3.0"', 'multiprocessing;python_version<"3.0"'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
