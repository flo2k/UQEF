"""
Abstract base class for a solver that manages the work.

@author: Florian Kuenzner
"""

from .SolverTimes import *
from .. import schedule

import numpy as np
from abc import abstractmethod
from abc import ABCMeta

# time measure
import time


class Solver(object):
    """
    Abstract solver interface
    """

    __metaclass__ = ABCMeta  # declare as abstract class

    def __init__(self):
        self.results = np.array([])
        self._timesteps = np.array([])

        self.solverTimes = SolverTimes()
        pass

    @abstractmethod
    def getSetup(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def init(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def tearDown(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def prepare(self, parameters):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def solve(self, runtime_estimator=None, chunksize=1,
              algorithm=schedule.Algorithm.FCFS, strategy=schedule.Strategy.FIXED_LINEAR):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def _assertParameters(self, parameters):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def _normaliseParameters(self, parameters):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def timesteps(self):
        raise NotImplementedError("Should have implemented this")

    def _estimateWorkRuntime(self, work_parameters, runtime_estimator):
        if runtime_estimator:
            estimated_runtimes = [runtime_estimator(*p) for p in work_parameters]
        else:
            estimated_runtimes = [1] * len(work_parameters)

        estimated_runtimes = np.asarray(estimated_runtimes)

        return estimated_runtimes

    def calcExecutionOrder(self, runtime_estimator=None):
        if runtime_estimator is None:
            self.sorted_indexes = range(len(self.parameters))
            self.original_indexes = range(len(self.parameters))
        else:
            print("start runtime estimation for calc execution order...")
            estimation_time_start = time.time()

            #paralleliser = Parallel(n_jobs=2, verbose=0, backend="threading")
            #estimated_runtimes = paralleliser(delayed(runtime_estimator)(*p) for p in self.parameters)
            estimated_runtimes = [runtime_estimator(*p) for p in self.parameters]

            estimation_time_end = time.time()
            print("estimation takes: {} s".format(estimation_time_end - estimation_time_start))

            self.sorted_indexes = sorted(range(len(estimated_runtimes)), key=lambda k: estimated_runtimes[k])
            self.original_indexes = sorted(range(len(self.sorted_indexes)), key=lambda k: self.sorted_indexes[k])

        return (self.sorted_indexes, self.original_indexes)

    def _sortParameters(self, parameters, sorted_indexes):
        return [parameters[i] for i in sorted_indexes]

    def _undoSortResults(self, results, original_indexes):
        return [results[i] for i in original_indexes]

    @abstractmethod
    def estimate_TProp(self, runtime_estimator=None, chunksize=1):
        raise NotImplementedError("Should have implemented this")

