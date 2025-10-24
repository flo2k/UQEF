"""
MpiSolver uses traditional MPI mechanism (scatter/gather) to distribute the work and collect the results. It also
supports a hybrid approach with combined thread parallelisation.

@author: Florian Kuenzner
"""


from .Solver import Solver
from .SolverTimes import *
from .. import schedule

from mpi4py import MPI

# for parallel computing
import multiprocessing

import numpy as np
import more_itertools

from joblib import Parallel, delayed

# time measure
import time


def _parallelSolve(model_generator, i_s, p_s):
    #print "i_s: " + str(i_s)
    #print "p_s: " + str(p_s)

    model = model_generator()
    model.prepare()

    #     rank = MPI.COMM_WORLD.Get_rank()
    #     print "rank: " + str(rank) + " runs: " + str(i)

    result = model.run(i_s, p_s)

    # print "results: " + str(np.asarray(result).shape)

    return result


class MpiSolver(Solver):
    """
    MpiSolver solves the work packages in parallel using a MPI
    """

    def __init__(self, model_generator, mpi_chunksize=1, unordered=False, normaliseParams=False, combinedParallel=False, num_cores=1):
        Solver.__init__(self)

        # behavior
        self.model_generator = model_generator
        self.mpi_chunksize = mpi_chunksize
        self.unordered = unordered
        self.normaliseParams = normaliseParams
        self.combinedParallel = combinedParallel

        self.size = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.name = MPI.Get_processor_name()
        self.version = MPI.Get_library_version()

        if self.rank == 0:
            self.infoModel = model_generator()

        self.numCores = num_cores
        print("rank {} uses numCores: {}".format(self.rank, self.numCores))

        self.solverTimes.num_work_packages = self.size
        self.solverTimes.parallel_solvers_per_work_package = np.array([self.numCores] * self.solverTimes.num_work_packages)

    def getSetup(self):
        return "%s using %d num mpi processes (mpi_chunksize=%d) with each %d cores" % (type(self).__name__, self.size, self.mpi_chunksize, self.numCores)

    def init(self):
        pass

    def tearDown(self):
        pass

    def prepare(self, parameters):
        self.parameters = parameters
        if self.rank == 0:
            self.infoModel.prepare()

    def solve(self, runtime_estimator=None, chunksize=1,
              algorithm=schedule.Algorithm.FCFS, strategy=schedule.Strategy.FIXED_LINEAR):
        if self.rank == 0:
            work_parameters = self.parameters
            # assert
            self._assertParameters(work_parameters)
            work_parameters = self._normaliseParameters(work_parameters)
            self._assertParameters(work_parameters)

            t_estimate_runtime_start = time.time()

            # estimate work runtime
            estimated_runtimes = self._estimateWorkRuntime(work_parameters, runtime_estimator)

            t_estimate_runtime_end = time.time()
            t_estimate_runtime = t_estimate_runtime_end - t_estimate_runtime_start
            print("t_estimate_runtime: {}".format(t_estimate_runtime))

            t_wp_creation_start = time.time()

            # generate work packages
            self.work_package_indexes = schedule.generate_work_package(estimated_runtimes, self.size,
                                                                       algorithm, strategy)

            # generate chunks and ensure to be able to restore the original order
            if strategy == schedule.Strategy.DYNAMIC:
                raise NotImplementedError("Strategy.DYNAMIC not supported by MpiSolver!")
            elif strategy in [schedule.Strategy.FIXED_ALTERNATE, schedule.Strategy.FIXED_LINEAR]:
                self.solverTimes.num_work_packages = len(self.work_package_indexes)
                self.solverTimes.parallel_solvers_per_work_package = np.array([1] * self.size)

                i_s_chunk = self.work_package_indexes
                parameterChunks = [work_parameters[i_s] for i_s in i_s_chunk]
                chunks = zip(i_s_chunk, parameterChunks)
                dataToSend = list(chunks)

            sorted_indexes = []
            for i_s_c in i_s_chunk:
                for i_s in i_s_c:
                    sorted_indexes.append(i_s)
            original_indexes = sorted(range(len(sorted_indexes)), key=lambda k: sorted_indexes[k])

            t_wp_creation_end = time.time()
            t_wp_creation = t_wp_creation_end - t_wp_creation_start
            print("t_wp_creation: {}".format(t_wp_creation))

        else:
            dataToSend = None

        if self.rank == 0: solver_time_start = time.time()

        #scatter
        if self.rank == 0: scatter_time_start = time.time()
        #if self.rank == 0: print("data to send: {}".format(dataToSend))
        data = MPI.COMM_WORLD.scatter(sendobj=dataToSend, root=0)
        if self.rank == 0: scatter_time_end = time.time()

        i_s, nodes = data
        nodes = np.array(nodes)

        print("rank: {}".format(self.rank))
        print("number of nodes: {}".format(nodes.T.shape))
        #print "nodes: " + str(nodes.T)

        if self.combinedParallel:
            chunk_results = Parallel(n_jobs=self.numCores, verbose=5, backend="threading")(
                delayed(_parallelSolve)(self.model_generator, [i_s], [p_s]) for i_s, p_s in zip(i_s, nodes))
        else:
            chunk_results = [_parallelSolve(self.model_generator, [i_s], [p_s]) for i_s, p_s in zip(i_s, nodes)]

        # gather
        if self.rank == 0: gather_time_start = time.time()
        chunk_results = MPI.COMM_WORLD.gather(sendobj=chunk_results)
        if self.rank == 0: gather_time_end = time.time()

        if self.rank == 0: solver_time_end = time.time()

        if self.rank == 0:
            results = []
            runtimes = []
            for chunk_result in chunk_results:
                for result in chunk_result:
                    for r in result:
                        results.append(r[0])
                        runtimes.append(r[1])

            t_estimate_restore_order_start = time.time()

            # restore initial order
            results = self._undoSortResults(results, original_indexes)
            runtimes = self._undoSortResults(runtimes, original_indexes)

            t_estimate_restore_order_end = time.time()
            t_estimate_restore_order = t_estimate_restore_order_end - t_estimate_restore_order_start
            print("t_estimate_restore_order: {}".format(t_estimate_restore_order))
            sys.stdout.flush()

            self.solverTimes.t_estimate_runtime = t_estimate_runtime
            self.solverTimes.t_wp_creation = t_wp_creation
            self.solverTimes.t_estimate_restore_order = t_estimate_restore_order

            # remember results
            self.results = results
            #print "results: " + str(np.array(results, dtype=object).shape)
            #print "results: " + str(self.results)
            # self._timesteps = self.infoModel.timesteps()

        if self.rank == 0:
            #scatter_time = scatter_time_end - scatter_time_start
            #gather_time = gather_time_end - gather_time_start
            #self.solverTimes.T_C = scatter_time + gather_time
            #self.solverTimes.T_C = scatter_time + gather_time

            solver_time = solver_time_end - solver_time_start
            #solver_time -= self.solverTimes.T_C
            print("xx solver_time: {}".format(solver_time))
            sys.stdout.flush()

            self.solverTimes.T_i_S = np.array(runtimes)

            self.solverTimes.T_i_SWP_i_worker = []
            for wp in self.work_package_indexes:
                self.solverTimes.T_i_SWP_i_worker.append([self.solverTimes.T_i_S[wi] for wi in wp])

            self.solverTimes.T_i_SWP_worker = np.zeros(len(self.work_package_indexes))
            for i in range(0, len(self.solverTimes.T_i_SWP_worker)):
                self.solverTimes.T_i_SWP_worker[i] = np.sum(self.solverTimes.T_i_SWP_i_worker[i]) / \
                                                     self.solverTimes.parallel_solvers_per_work_package[i]
            self.solverTimes.T_SWP_worker = self.solverTimes.T_i_SWP_worker.max()

            self.solverTimes.T_S_overhead = float(0)

            self.solverTimes.T_i_I = calc_idle_of_work_packages(self.solverTimes.T_i_SWP_worker)
            self.solverTimes.T_I = self.solverTimes.T_i_I.max()

            self.solverTimes.T_C = solver_time - self.solverTimes.T_SWP_worker

            self.solverTimes.T_Prop = self.solverTimes.T_SWP_worker + self.solverTimes.T_S_overhead + self.solverTimes.T_C


    def _assertParameters(self, parameters):
        for parameter in parameters:
            self.infoModel.assertParameter(parameter)

    def _normaliseParameters(self, parameters):
        norm_paras = []
        for parameter in parameters:
            norm_para = self.infoModel.normaliseParameter(parameter)
            norm_paras.append(norm_para)

        return np.array(norm_paras)

    def estimate_TProp(self, runtime_estimator=None, chunksize=1):
        pass

    def timesteps(self):
        self._timesteps = self.infoModel.timesteps()
        return self._timesteps
