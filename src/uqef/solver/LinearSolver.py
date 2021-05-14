"""
Linear solver solves each "sample" one after the other - linearly - without any parallelisation.

@author: Florian Kuenzner
"""

from .Solver import Solver
from .SolverTimes import *
from .. import schedule

import numpy as np
import more_itertools

# time measure
import time


class LinearSolver(Solver):
    """
    LinearSolver solves one work package after another. There is no parallelisation involved.
    """

    def __init__(self, model_generator, normaliseParams=False):
        Solver.__init__(self)

        # behavior
        self.model_generator = model_generator
        self.normaliseParams = normaliseParams

        self.solverTimes.parallel_solvers_per_work_package = np.array([1])
        
        self.infoModel = model_generator()

    def getSetup(self):
        return "%s" % (type(self).__name__)

    def init(self):
        pass
    
    def tearDown(self):
        pass
        
    def prepare(self, parameters):
        self.parameters = parameters
        self.infoModel.prepare()
        
    def solve(self, runtime_estimator=None, chunksize=1,
              algorithm=schedule.Algorithm.FCFS, strategy=schedule.Strategy.FIXED_LINEAR):
        work_parameters = self.parameters
        # assert
        self._assertParameters(work_parameters)
        work_parameters = self._normaliseParameters(work_parameters)
        self._assertParameters(work_parameters)

        # sort work_parameters for optimal execution order
        #self.sorted_indexes, self.original_indexes = self.calcExecutionOrder(runtime_estimator)
        #work_parameters = self._sortParameters(work_parameters, self.sorted_indexes)

        # estimate work runtime
        estimated_runtimes = self._estimateWorkRuntime(work_parameters, runtime_estimator)

        # generate work packages
        self.work_package_indexes = schedule.generate_work_package(estimated_runtimes, 1, algorithm,
                                                                   strategy)

        # generate chunks and ensure to be able to restore the original order
        if strategy in [schedule.Strategy.DYNAMIC, schedule.Strategy.FIXED_LINEAR]:
            self.solverTimes.num_work_packages = len(self.work_package_indexes)
            self.solverTimes.parallel_solvers_per_work_package = np.array([1])

            sorted_indexes = schedule.generate_work_list_from_work_package(estimated_runtimes,
                                                                           self.work_package_indexes)
            work_parameters = self._sortParameters(work_parameters, sorted_indexes)

            # split into chunks
            i_s_chunk = list(more_itertools.chunked(sorted_indexes, chunksize))
            parameterChunks = list(more_itertools.chunked(work_parameters, chunksize))
            chunks = zip(i_s_chunk, parameterChunks)
        elif strategy in [schedule.Strategy.FIXED_ALTERNATE]:
            raise NotImplementedError("Strategy.FIXED_ALTERNATE not supported by LinearSolver!")

        sorted_indexes = []
        for i_s_c in i_s_chunk:
            for i_s in i_s_c:
                sorted_indexes.append(i_s)
        original_indexes = sorted(range(len(sorted_indexes)), key=lambda k: sorted_indexes[k])

        # do the simulation
        solver_time_start = time.time()
        solver_time2 = 0.0
        results = []
        runtimes = []
        for c in chunks:
            i_s, p_s = c
            solver_time_start2 = time.time()
            chunk_results = self.infoModel.run(i_s, p_s)
            solver_time_end2 = time.time()
            solver_time2 += (solver_time_end2 - solver_time_start2)
            for result in chunk_results:
                results.append(result[0])
                runtimes.append(result[1])

        solver_time_end = time.time()

        # calc timing
        solver_time = solver_time_end - solver_time_start

        #print("solver_time: {}".format(solver_time))
        #print("solver_time2: {}".format(solver_time2))
        solver_time = solver_time2

        # restore initial order
        results = self._undoSortResults(results, original_indexes)
        runtimes = self._undoSortResults(runtimes, original_indexes)

        self.solverTimes.T_i_S = np.array(runtimes)
        self.solverTimes.T_i_SWP_i_worker = []
        for wp in self.work_package_indexes:
            self.solverTimes.T_i_SWP_i_worker.append([self.solverTimes.T_i_S[wi] for wi in wp])

        self.solverTimes.T_i_SWP_worker = np.zeros(len(self.work_package_indexes))
        for i in range(0, len(self.solverTimes.T_i_SWP_i_worker)):
            self.solverTimes.T_i_SWP_worker[i] = np.sum(self.solverTimes.T_i_SWP_i_worker[i]) / \
                                                 self.solverTimes.parallel_solvers_per_work_package[i]
        self.solverTimes.T_SWP_worker = self.solverTimes.T_i_SWP_worker.max()

        self.solverTimes.T_S_overhead = float(0)

        self.solverTimes.T_i_I = calc_idle_of_work_packages(self.solverTimes.T_i_SWP_worker)
        self.solverTimes.T_I = self.solverTimes.T_i_I.max()

        # self.solverTimes.T_C = solver_time - self.solverTimes.T_SWP_worker
        self.solverTimes.T_C = 0

        self.solverTimes.T_Prop = self.solverTimes.T_SWP_worker + self.solverTimes.T_S_overhead + self.solverTimes.T_C

        # remember results
        self.results = results
        # self._timesteps = self.infoModel.timesteps()

    def _assertParameters(self, parameters):
        for parameter in parameters:
            self.infoModel.assertParameter(parameter)

    def _normaliseParameters(self, parameters):
        norm_paras = []
        for parameter in parameters:
            norm_para = self.infoModel.normaliseParameter(parameter)
            norm_paras.append(norm_para)

        return np.array(norm_paras)

    def timesteps(self):
        self._timesteps = self.infoModel.timesteps()
        return self._timesteps

    def estimate_TProp(self, runtime_estimator=None, chunksize=1):
        pass