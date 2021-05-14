"""
Parallel solver uses a thread pool to distribute the work across all cores on a node dynamically.

@author: Florian Kuenzner
"""

from .Solver import Solver
from .SolverTimes import *
from .. import schedule

import numpy as np
import more_itertools
from joblib import Parallel, delayed

# time measure
import time


def _parallelSolve(model_generator, i_s, p_s):
    model = model_generator()
    model.prepare()

    results = model.run(i_s, p_s)
    return results


def _parallelSolve_for_measure_overhead(i, sleep_time):
    #time.sleep(sleep_time)
    return i


class ParallelSolver(Solver):
    """
    ParallelSolver solves the work packages in parallel using a pool from joblib
    """

    def __init__(self, model_generator, numCores, normaliseParams=False):
        Solver.__init__(self)

        # behavior
        self.model_generator = model_generator
        self.numCores = numCores
        self.normaliseParams = normaliseParams

        self.solverTimes.num_work_packages = 1
        self.solverTimes.parallel_solvers_per_work_package = np.array([self.numCores])

        self.infoModel = model_generator()
        
    def getSetup(self):
        return "%s using %d cores" % (type(self).__name__, self.numCores)

    def init(self):
        pass

    def tearDown(self):
        pass

    def prepare(self, parameters):
        self.parameters = parameters
        self.infoModel.prepare()

    def solve(self, runtime_estimator=None, chunksize=1,
              algorithm=schedule.Algorithm.FCFS, strategy=schedule.Strategy.DYNAMIC):
        work_parameters = self.parameters

        # assert
        self._assertParameters(work_parameters)
        work_parameters = self._normaliseParameters(work_parameters)
        self._assertParameters(work_parameters)

        # sort work_parameters for optimal execution order
        #self.sorted_indexes, self.original_indexes = self.calcExecutionOrder(runtime_estimator)
        #work_parameters = self._sortParameters(work_parameters, self.sorted_indexes)

        t_estimate_runtime_start = time.time()

        # estimate work runtime
        estimated_runtimes = self._estimateWorkRuntime(work_parameters, runtime_estimator)

        t_estimate_runtime_end = time.time()
        t_estimate_runtime = t_estimate_runtime_end - t_estimate_runtime_start
        print("t_estimate_runtime: {}".format(t_estimate_runtime))

        t_wp_creation_start = time.time()

        # generate work packages
        self.work_package_indexes = schedule.generate_work_package(estimated_runtimes, self.numCores, algorithm, strategy)

        # generate chunks and ensure to be able to restore the original order
        if strategy == schedule.Strategy.DYNAMIC:
            self.solverTimes.num_work_packages = self.numCores
            self.solverTimes.parallel_solvers_per_work_package = np.array([1] * self.numCores)

            sorted_indexes = schedule.generate_work_list_from_work_package(estimated_runtimes,
                                                                           self.work_package_indexes)
            #original_indexes = sorted(range(len(self.sorted_indexes)), key=lambda k: self.sorted_indexes[k])
            work_parameters = self._sortParameters(work_parameters, sorted_indexes)
            #self.work_package_indexes = [self.sorted_indexes]  # here we only have one work package

            # split into chunks
            i_s_chunk = list(more_itertools.chunked(sorted_indexes, chunksize))
            parameterChunks = list(more_itertools.chunked(work_parameters, chunksize))
            chunks = zip(i_s_chunk, parameterChunks)
        elif strategy in [schedule.Strategy.FIXED_ALTERNATE, schedule.Strategy.FIXED_LINEAR]:
            self.solverTimes.num_work_packages = len(self.work_package_indexes)
            self.solverTimes.parallel_solvers_per_work_package = np.array([1]*self.numCores)

            i_s_chunk = self.work_package_indexes
            parameterChunks = [work_parameters[i_s] for i_s in i_s_chunk]
            chunks = zip(i_s_chunk, parameterChunks)

        sorted_indexes = []
        for i_s_c in i_s_chunk:
            for i_s in i_s_c:
                sorted_indexes.append(i_s)
        original_indexes = sorted(range(len(sorted_indexes)), key=lambda k: sorted_indexes[k])

        t_wp_creation_end = time.time()
        t_wp_creation = t_wp_creation_end - t_wp_creation_start
        print("t_wp_creation: {}".format(t_wp_creation))

        # init paralleliser
        paralleliser = Parallel(n_jobs=self.numCores, verbose=0, backend="threading")

        # measure threading overhead
        overhead_time_start = time.time()
        overhead_results = paralleliser(delayed(_parallelSolve_for_measure_overhead)(_, 0) for _ in range(0, len(work_parameters)))
        overhead_time_end = time.time()

        # do the simulation
        paralleliser = Parallel(n_jobs=self.numCores, verbose=5, backend="threading")
        solver_time_start = time.time()
        chunk_results = paralleliser(delayed(_parallelSolve)(self.model_generator, i_s, p_s)
                               for (i_s, p_s) in chunks)
        solver_time_end = time.time()

        # results
        results = []
        runtimes = []
        for chunk_result in chunk_results:
            for result in chunk_result:
                results.append(result[0])
                runtimes.append(result[1])

        # calc timing
        overhead_time = overhead_time_end - overhead_time_start #the overhead time is the time of overhead for joblib parallelisation
        solver_time = solver_time_end - solver_time_start
        #solver_time -= overhead_time

        print("solver_time: {}".format(solver_time))
        print("overhead_time: {}".format(overhead_time))

        t_estimate_restore_order_start = time.time()

        # restore initial order
        results = self._undoSortResults(results, original_indexes)
        runtimes = self._undoSortResults(runtimes, original_indexes)

        t_estimate_restore_order_end = time.time()
        t_estimate_restore_order = t_estimate_restore_order_end - t_estimate_restore_order_start
        print("t_estimate_restore_order: {}".format(t_estimate_restore_order))

        self.solverTimes.t_estimate_runtime = t_estimate_runtime
        self.solverTimes.t_wp_creation = t_wp_creation
        self.solverTimes.t_estimate_restore_order = t_estimate_restore_order

        self.solverTimes.T_i_S = np.array(runtimes)

        self.solverTimes.T_i_SWP_i_worker = []
        for wp in self.work_package_indexes:
            self.solverTimes.T_i_SWP_i_worker.append([self.solverTimes.T_i_S[wi] for wi in wp])

        self.solverTimes.T_i_SWP_worker = np.zeros(len(self.work_package_indexes))
        for i in range(0, len(self.solverTimes.T_i_SWP_i_worker)):
            self.solverTimes.T_i_SWP_worker[i] = np.sum(self.solverTimes.T_i_SWP_i_worker[i]) / self.solverTimes.parallel_solvers_per_work_package[i]
        self.solverTimes.T_SWP_worker = self.solverTimes.T_i_SWP_worker.max()

        self.solverTimes.T_S_overhead = float(0)

        self.solverTimes.T_i_I = calc_idle_of_work_packages(self.solverTimes.T_i_SWP_worker)
        self.solverTimes.T_I = self.solverTimes.T_i_I.max()

        #self.solverTimes.T_C = solver_time - self.solverTimes.T_SWP_worker
        self.solverTimes.T_C = 0

        self.solverTimes.T_Prop = self.solverTimes.T_SWP_worker + self.solverTimes.T_S_overhead + self.solverTimes.T_C



        #remember results
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