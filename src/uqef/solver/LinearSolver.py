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

    def __init__(self, modelGenerator, normaliseParams=False):
        LinearSolver.__init__(self)

        # behavior
        self.modelGenerator = modelGenerator
        self.normaliseParams = normaliseParams

        self.solverTimes.parallel_solvers_per_work_package = np.array([1])
        
        self.infoModel = modelGenerator()

    def getSetup(self):
        return "%s" % (type(self).__name__)

    def init(self):
        pass
    
    def tearDown(self):
        pass
        
    def prepare(self, parameters):
        self.parameters = parameters
        self.infoModel.prepare()
        
    def solve(self, runtime_estimator, chunksize):
        work_parameters = self.parameters
        # assert
        self._assertParameters(work_parameters)
        work_parameters = self._normaliseParameters(work_parameters)
        self._assertParameters(work_parameters)

        # sort work_parameters for optimal execution order
        self.sorted_indexes, self.original_indexes = self.calcExecutionOrder(runtime_estimator)
        work_parameters = self._sortParameters(work_parameters, self.sorted_indexes)

        # split into chunks
        i_s_chunk = list(more_itertools.chunked(range(0, len(work_parameters)), chunksize))
        parmeterChunks = list(more_itertools.chunked(work_parameters, chunksize))
        chunks = zip(i_s_chunk, parmeterChunks)

        # do the simulation
        solver_time_start = time.time()
        solver_time2 = 0.0
        results = []
        for c in chunks:
            i_s, p_s = c
            solver_time_start2 = time.time()
            chunk_results = self.infoModel.run(i_s, p_s)
            solver_time_end2 = time.time()
            solver_time2 += (solver_time_end2 - solver_time_start2)
            for result in chunk_results:
                results.append(result)

        solver_time_end = time.time()

        # calc timing
        solver_time = solver_time_end - solver_time_start

        print("solver_time: {}".format(solver_time))
        print("solver_time2: {}".format(solver_time2))
        solver_time = solver_time2

        self.solverTimes.T_i_S = np.array(results)
        self.solverTimes.T_i_SWP_i_worker = np.zeros((1, len(self.solverTimes.T_i_S)))
        self.solverTimes.T_i_SWP_i_worker[0] = self.solverTimes.T_i_S
        self.solverTimes.T_i_SWP_worker = np.array([solver_time])

        self.solverTimes.T_SWP_worker = self.solverTimes.T_i_SWP_worker.sum()
        self.solverTimes.T_Prop = solver_time

        self.solverTimes.T_i_I = np.array(0)
        self.solverTimes.T_I = self.solverTimes.T_Prop - self.solverTimes.T_i_S.sum()

        self.solverTimes.T_C = 0

        # restore initial order
        results = self._undoSortResults(results, self.original_indexes)

        # remember results
        self.results = results
        self.timesteps = self.infoModel.timesteps()

    def _assertParameters(self, parameters):
        for parameter in parameters:
            self.infoModel.assertParameter(parameter)

    def _normaliseParameters(self, parameters):
        norm_paras = []
        for parameter in parameters:
            norm_para = self.infoModel.normaliseParameter(parameter)
            norm_paras.append(norm_para)

        return np.array(norm_paras)