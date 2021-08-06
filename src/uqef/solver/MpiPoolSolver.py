"""
MpiPoolSolver uses an MPI pool to distribute the work dynamically to all MPI processes.

It uses the mpi4py.future package with the new pool mechanism on MPI level. It is like a thread pool, but can distribute
the work across several nodes.

A possible drawback is: The MPI pool on rank 0 does not participate on the working process, it is reserved for
dynamically distribute the work to worker MPI processes and collecting the results.

@author: Florian Kuenzner
"""

from .Solver import Solver
from .SolverTimes import *
from .. import schedule

from mpi4py import MPI
import mpi4py.futures as futures

# for parallel computing
import multiprocessing

import numpy as np
import more_itertools

from joblib import Parallel, delayed


def _parallelSolve(model_generator, i_s, p_s):
    # print "i_s: " + str(i_s)
    # print "p_s: " + str(p_s)

    model = model_generator()
    model.prepare()

    # rank = MPI.COMM_WORLD.Get_rank()
    # print (f"rank: {rank} runs: {i}")

    result = model.run(i_s, p_s)

    # print (f"results: {np.asarray(result).shape}")

    return result


def _combinedParallelSolve(model_generator, i_s, p_s, num_cores):
    # print "c_i_s: " + str(i_s)
    # print "c_p_s: " + str(p_s)

    # num_cores = multiprocessing.cpu_count()
    #    parallelSolver = ParallelSolver(model_generator, num_cores)

    paralleliser = Parallel(n_jobs=num_cores, verbose=5)
    results = paralleliser(delayed(_parallelSolve)(model_generator, [i], [p]) for (i, p) in zip(i_s, p_s))

    # print "results: " + str(np.asarray(results).shape)

    #    parallelSolver.solve()

    return results


class MpiPoolSolver(Solver):
    """
    MpiPoolSolver solves the work packages in parallel using a MPI pool
    """

    def __init__(self, model_generator, mpi_chunksize=1, unordered=False, normaliseParams=False, combinedParallel=False,
                 num_cores=1):
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

        if self.combinedParallel:
            self.parallel_solvers_per_work_package = np.array([num_cores] * self.size)
        else:
            self.parallel_solvers_per_work_package = np.ones(self.size)

    def getSetup(self):
        return "%s using %d num mpi processes (mpi_chunksize=%d) with each %d cores" % (
        type(self).__name__, self.size, self.mpi_chunksize, self.numCores)

    def init(self):
        pass

    def tearDown(self):
        pass

    def prepare(self, parameters):
        self.parameters = parameters
        if self.rank == 0:
            self.infoModel.prepare(infoModel=True)

    def solve(self, runtime_estimator=None, chunksize=1,
              algorithm=schedule.Algorithm.FCFS, strategy=schedule.Strategy.DYNAMIC):
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
                self.solverTimes.num_work_packages = len(self.work_package_indexes)
                self.solverTimes.parallel_solvers_per_work_package = np.array([1] * self.size)

                sorted_indexes = schedule.generate_work_list_from_work_package(estimated_runtimes,
                                                                               self.work_package_indexes)
                # original_indexes = sorted(range(len(self.sorted_indexes)), key=lambda k: self.sorted_indexes[k])
                work_parameters = self._sortParameters(work_parameters, sorted_indexes)
                # self.work_package_indexes = [self.sorted_indexes]  # here we only have one work package

                # split into chunks
                i_s_chunk = list(more_itertools.chunked(sorted_indexes, chunksize))
                parameterChunks = list(more_itertools.chunked(work_parameters, chunksize))
                chunks = zip(i_s_chunk, parameterChunks)

                # i_s_chunk = self.work_package_indexes
                # parameterChunks = [work_parameters[i_s] for i_s in i_s_chunk]
                # chunks = zip(i_s_chunk, parameterChunks)
                # dataToSend = chunks
                # print(len(dataToSend))

            elif strategy in [schedule.Strategy.FIXED_ALTERNATE, schedule.Strategy.FIXED_LINEAR]:
                raise NotImplementedError(
                    "Strategy.FIXED_ALTERNATE or Strategy.FIXED_LINEAR not supported by MpiPoolSolver!")

            sorted_indexes = []
            for i_s_c in i_s_chunk:
                for i_s in i_s_c:
                    sorted_indexes.append(i_s)
            original_indexes = sorted(range(len(sorted_indexes)), key=lambda k: sorted_indexes[k])

            t_wp_creation_end = time.time()
            t_wp_creation = t_wp_creation_end - t_wp_creation_start
            print("t_wp_creation: {}".format(t_wp_creation))

        # do the simulation
        with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:  # master process
                solver_time_start = time.time()

                if self.combinedParallel == False:
                    chunk_results_it = executor.map(_parallelSolve, [self.model_generator] * len(i_s_chunk), i_s_chunk,
                                                    parameterChunks,
                                                    chunksize=self.mpi_chunksize, unordered=self.unordered)
                else:
                    chunk_results_it = executor.map(_combinedParallelSolve, [self.model_generator] * len(i_s_chunk),
                                                    i_s_chunk,
                                                    parameterChunks, [self.numCores] * len(i_s_chunk),
                                                    chunksize=self.mpi_chunksize, unordered=self.unordered)

                print("{}: waits for shutdown...".format(self.rank))
                sys.stdout.flush()
                executor.shutdown(wait=True)
                print("{}: shutted down...".format(self.rank))
                sys.stdout.flush()

                solver_time_end = time.time()

                # print "chunk_results: " + str(chunk_results)
                chunk_results = list(chunk_results_it)

                results = []
                runtimes = []
                for chunk_result in chunk_results:
                    for result in chunk_result:
                        if self.combinedParallel:
                            for r in result:
                                results.append(r[0])
                                runtimes.append(r[1])
                        else:
                            results.append(result[0])
                            runtimes.append(result[1])

                #                 for r in results:
                #                     print "result = " + str(r)

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

                self.results = results
                # print "results: " + str(np.array(results, dtype=object).shape)
                # print "results: " + str(self.results)
                # self._timesteps = self.infoModel.timesteps()

                solver_time = solver_time_end - solver_time_start
                # solver_time -= self.solverTimes.T_C
                print("xx solver_time: {}".format(solver_time))
                sys.stdout.flush()
                # solver times
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
        if self.rank == 0:
            for parameter in parameters:
                self.infoModel.assertParameter(parameter)

    def _normaliseParameters(self, parameters):
        norm_paras = []
        if self.rank == 0:
            for parameter in parameters:
                norm_para = self.infoModel.normaliseParameter(parameter)
                norm_paras.append(norm_para)

        return np.array(norm_paras)

    def estimate_TProp(self, runtime_estimator=None, chunksize=1):
        pass

    def timesteps(self):
        if self.rank == 0:
            self._timesteps = self.infoModel.timesteps()
        return self._timesteps
