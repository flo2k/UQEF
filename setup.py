#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='uqef',
    version="1.0.0",
    url='https://gitlab.lrz.de/tum-i05/software/UQEF',
    author="Florian Künzner",
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
        'more-itertools',
        'mpi4py',
        'numpy',
        'psutil',
        'scikit-learn',
        'scipy',
        'seaborn',
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
