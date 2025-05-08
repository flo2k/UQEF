#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='uqef',
    version="0.4",
    url='https://gitlab.lrz.de/tum-i05/software/UQEF',
    author="Florian Kuenzner",
    author_email='florian.kuenzner@tum.de',
    license='MIT',
    platforms='any',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        'chaospy',
        'dill',
        'joblib',
        'matplotlib',
        'more_itertools',
        'mpi4py',
        'numpy',
        'scipy',
        'setuptools',
        'tabulate'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    ],
)
