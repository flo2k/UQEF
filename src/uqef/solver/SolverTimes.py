"""
SolverTimes is a helper class to collect the different measured runtime aspects.

@author: Florian Kuenzner
"""

import numpy as np
import itertools
import more_itertools
import sys
import scipy.optimize
from enum import Enum
import time


class SolverTimes(object):
    """
    SolverTimes is a container for the runtimes of the solver
    """

    def __init__(self):
        self.num_work_packages = 1
        self.parallel_solvers_per_work_package = np.array([])

        #per solver
        self.T_i_S = np.array([])

        #solver time per work package
        self.T_i_SWP_i_worker = np.zeros((0,0))
        self.T_i_SWP_worker = np.zeros(0)
        self.T_SWP_worker = float(0.0)

        #overhead of the solver
        self.T_S_overhead = float(0.0)
        self.t_estimate_runtime = float(0.0)
        self.t_wp_creation = float(0.0)
        self.t_estimate_restore_order = float(0.0)

        #propagation
        self.T_Prop = float(0.0)

        #idle
        self.T_i_I = np.zeros(0)
        self.T_I = float(0.0)

        #communication
        self.T_C = float(0.0)

        pass


def sim_runtime_of_work_package(work, num_worker):
    working = np.zeros(num_worker)
    worker_solver_time = np.full(num_worker, np.nan)
    worker_start_time = np.zeros(num_worker)
    worker_end_time = np.full(num_worker, np.nan)

    work_todo = list(work)
    current_time = 0.0

    while len(work_todo) != 0 or working.any():
        #start work
        for iw in range(0, len(working)):
            if working[iw] == 0 and len(work_todo) > 0:
                working[iw] = 1
                worker_solver_time[iw] = work_todo.pop(0)
                worker_start_time[iw] = current_time
                worker_end_time[iw] = worker_start_time[iw] + worker_solver_time[iw]

        #stop work
        if working.any():
            worker_index = np.nanargmin(worker_end_time)
            current_time = worker_end_time[worker_index]

            #reset worker
            working[worker_index] = 0
            worker_solver_time[worker_index] = np.nan
            worker_start_time[worker_index] = 0
            worker_end_time[worker_index] = np.nan

    runtime = current_time
    return runtime


def sim_runtime_of_work_package_heuristic(work, num_worker):
    worker_work = [[] for _ in range(0, num_worker)]
    optimal_runtime = np.max(work)/num_worker

    work_todo = list(work)

    #first iteration!
    while len(work_todo) != 0:
        for i_w in range(0, num_worker):
            if len(work_todo) > 0:
                worker_work[i_w].append(work_todo.pop(0))

            while len(work_todo) > 0 and np.sum(worker_work[i_w]) + work_todo[0] < optimal_runtime:
                worker_work[i_w].append(work_todo.pop(0))

#    if len(work_todo) != 0:
#        for i_w in range(0, num_worker):
#            if len(worker_work[i_w]) == 0 and len(work_todo) > 0:
#                worker_work[i_w].append(work_todo.pop(0))#

            #while np.sum(worker_work[i_w]) + work_todo[0] < optimal_runtime:
             #   worker_work[i_w].append(work_todo.pop(0))

    runtime = np.max([np.sum(_) if len(_) != 0 else sys.maxint for _ in worker_work])

    return (runtime, worker_work)


#SimRuntimeStrategy = Enum('minimum', 'maximum', 'asitis')
class SimRuntimeStrategy(Enum):
    minimum = 1
    maximum = 2
    asitis  = 3

def sim_runtime_of_work_package2(work, num_worker, strategy=SimRuntimeStrategy.minimum):
    working = np.zeros(num_worker)
    worker_solver_time = np.full(num_worker, np.nan)
    worker_start_time = np.zeros(num_worker)
    worker_end_time = np.full(num_worker, np.nan)
    worker_work = [[] for _ in range(0, num_worker)]

    work_todo = list(work)
    current_time = 0.0

    while len(work_todo) != 0 or working.any():
        # start work
        for iw in range(0, len(working)):
            if working[iw] == 0 and len(work_todo) > 0:
                working[iw] = 1
                if strategy in [SimRuntimeStrategy.minimum, SimRuntimeStrategy.asitis]:
                    worker_solver_time[iw] = work_todo.pop(0)
                elif strategy == SimRuntimeStrategy.maximum:
                    worker_solver_time[iw] = work_todo.pop(0 if iw == 0 else -1) #worker 0 get the longest, the other the shortest
                    #worker_solver_time[iw] = work_todo.pop(-1)  # worker 0 get the longest, the other the shortest

                worker_work[iw].append(worker_solver_time[iw])
                worker_start_time[iw] = current_time
                worker_end_time[iw] = worker_start_time[iw] + worker_solver_time[iw]

        # stop work
        if working.any():
            worker_index = np.nanargmin(worker_end_time)
            current_time = worker_end_time[worker_index]

            # reset worker
            working[worker_index] = 0
            worker_solver_time[worker_index] = np.nan
            worker_start_time[worker_index] = 0
            worker_end_time[worker_index] = np.nan

    runtime = current_time
    return (runtime, worker_work)


def sim_runtime_of_work_packages(work, parallel_solvers_per_work_package):
    runtime = np.asarray([sim_runtime_of_work_package(w, nw) for (w, nw) in zip(work, parallel_solvers_per_work_package)])
    return runtime


def sim_runtime_of_work_packages2(work_packages, parallel_solvers_per_work_package, strategy=SimRuntimeStrategy.minimum):
    runtime_data = [sim_runtime_of_work_package2(wp, nw, strategy)
                          for (wp, nw)
                          in zip(work_packages, parallel_solvers_per_work_package)]

    wp_runtimes = []
    wp_work = []
    for data in runtime_data:
        wp_runtime, wp_ = data
        wp_runtimes.append(wp_runtime)
        wp_work.append(wp_)

    return (max(wp_runtimes), wp_runtimes, wp_work)


def create_work_package(work, num_work_packages, strategy=SimRuntimeStrategy.minimum):
    if strategy in [SimRuntimeStrategy.maximum, SimRuntimeStrategy.asitis]:
        work_package_it = list(more_itertools.divide(num_work_packages, work))
    elif strategy == SimRuntimeStrategy.minimum:
        work_package_it = list(more_itertools.distribute(num_work_packages, work))

    work_packages = [list(wp) for wp in work_package_it]
    return work_packages


def find_optimal_and_worst_runtimes(work, num_workpackages, parallel_solvers_per_work_package):
#    print("w1")
    work_combinations = list(more_itertools.distinct_permutations(work))
#    print("len: " + str(len(work)))
#    print("len: " + str(len(work_combinations)))
#    print("w2")
    work_combinations_work_packages = [create_work_package(w, num_workpackages) for w in work_combinations]

#    print("w3")
    runtimes_combinations_results = [sim_runtime_of_work_packages2(wps, parallel_solvers_per_work_package) for wps in work_combinations_work_packages]
#    print("w4")
    runtimes_combinations_work = [r[0] for r in runtimes_combinations_results]
#    print "runtimes_combinations_work: " + str(runtimes_combinations_work)
#    print("w5")
    index_min_runtime = np.argmin(runtimes_combinations_work)
#    print("w6")
    index_max_runtime = np.argmax(runtimes_combinations_work)

#    print("w7")
    min_runtime = (runtimes_combinations_results[index_min_runtime][0], runtimes_combinations_results[index_min_runtime][1], work_combinations[index_min_runtime], work_combinations_work_packages[index_min_runtime], runtimes_combinations_results[index_min_runtime][2])
    max_runtime = (runtimes_combinations_results[index_max_runtime][0], runtimes_combinations_results[index_max_runtime][1], work_combinations[index_max_runtime], work_combinations_work_packages[index_max_runtime], runtimes_combinations_results[index_max_runtime][2])
#    print("w8")
    return (min_runtime, max_runtime)


def find_optimal_and_worst_runtimes1(work, num_workpackages, parallel_solvers_per_work_package):
    work_combinations_it = list(more_itertools.distinct_permutations(work))

    min_runtime = sys.float_info.max
    max_runtime = 0.0

    min_runtime_work = []
    max_runtime_work = []

    min_runtime_work_package = []
    max_runtime_work_package = []

    for work_combination in work_combinations_it:
        work_package = create_work_package(work_combination, num_workpackages)

        runtimes_combinations_work_package = sim_runtime_of_work_packages(work_package, parallel_solvers_per_work_package)
        runtimes_combinations_work = runtimes_combinations_work_package.max()

        if(runtimes_combinations_work < min_runtime):
            min_runtime = runtimes_combinations_work
            min_runtime_work = work_combination
            min_runtime_work_package = work_package

        if (runtimes_combinations_work > max_runtime):
            max_runtime = runtimes_combinations_work
            max_runtime_work = work_combination
            max_runtime_work_package = work_package

    min_runtime_t = (min_runtime, min_runtime_work, min_runtime_work_package)
    max_runtime_t = (max_runtime, max_runtime_work, max_runtime_work_package)

    return (min_runtime_t, max_runtime_t)


def find_optimal_and_worst_runtimes2(work, num_workpackages, parallel_solvers_per_work_package):
    min_work = np.sort(work)[::-1].copy()
    min_work_packages = create_work_package(min_work, num_workpackages, SimRuntimeStrategy.minimum)
    min_work_runtime, min_work_wp = sim_runtime_of_work_packages2(min_work_packages, parallel_solvers_per_work_package, SimRuntimeStrategy.minimum)

    max_work = np.sort(work)[::-1].copy()
    max_work_packages = create_work_package(max_work, num_workpackages, SimRuntimeStrategy.maximum)
    max_work_runtime, max_work_wp = sim_runtime_of_work_packages2(max_work_packages, parallel_solvers_per_work_package, SimRuntimeStrategy.maximum)

    print("min_runtime: {}".format(min_work_runtime))
    print("max_runtime: {}".format(max_work_runtime))

    min_runtime = (min_work_runtime, min_work, min_work_wp)
    max_runtime = (max_work_runtime, max_work, max_work_wp)
    return (min_runtime, max_runtime)


def find_optimal_and_worst_runtimes3(work, num_workpackages, parallel_solvers_per_work_package):
    min_work = np.sort(work)[::-1].copy()
    min_work_packages = create_work_package(min_work, num_workpackages, SimRuntimeStrategy.minimum)
    min_runtime_results = sim_runtime_of_work_packages2(min_work_packages, parallel_solvers_per_work_package, SimRuntimeStrategy.minimum)
    min_work_runtime, min_work_runtimes, min_work_wp = min_runtime_results

    max_work = np.sort(work)[::-1].copy()
    max_work_packages = create_work_package(max_work, num_workpackages, SimRuntimeStrategy.maximum)
    max_runtime_results = sim_runtime_of_work_packages2(max_work_packages, parallel_solvers_per_work_package, SimRuntimeStrategy.maximum)
    max_work_runtime, max_work_runtimes, max_work_wp = max_runtime_results

    min_data = (min_work_runtime, min_work_runtimes, min_work, min_work_packages, min_work_wp)
    max_data = (max_work_runtime, max_work_runtimes, max_work, max_work_packages, max_work_wp)
    return (min_data, max_data)


def calc_idle_of_work_packages(runtimes):
    idle = runtimes.max() - runtimes
    return idle

# TODO: remove the "__main__" function and write some UnitTests for it!
if __name__ == "__main__":
    # solverTimes = SolverTimes()
    #
    # work = np.asarray(range(0, 6))
    # #work = np.ones(3)
    # print("work: " + str(work))
    #
    # sim_time = sim_runtime_of_work_package(work, 2)
    # print "sim_time: " + str(sim_time)
    #
    # work = np.asarray([[1,1,1], [1,1,1,1,1]])
    # sim_time = sim_runtime_of_work_packages(work, [2, 2])
    # print "sim_time: " + str(sim_time)
    # print "idle_time: " + str(calc_idle_of_work_packages(sim_time))

    #work = np.asarray(range(0, 9))+1
#    work = np.array([0.47466246, 0.47466091, 0.47465936, 0.50062271, 0.50062115,
#                     0.5006196, 0.52658295, 0.52658139, 0.52657984, 0.4746358,
#                     0.47463425, 0.47463269, 0.50059604, 0.50059449, 0.50059293,
#                     0.52655628, 0.52655473, 0.52655318, 0.47460914, 0.47460758,
#                     0.47460603, 0.50056938, 0.50056782, 0.50056627, 0.52652962,
#                     0.52652806, 0.52652651])
#
    # work = np.array([ 0.52658295,  0.52658139,  0.52657984,  0.52655628,  0.52655473,
    #                   0.52655318,  0.52652962,  0.52652806,  0.52652651,  0.50062271,
    #                   0.50062115,  0.5006196 ,  0.50059604,  0.50059449,  0.50059293,
    #                   0.50056938,  0.50056782,  0.50056627,  0.47466246,  0.47466091,
    #                   0.47465936,  0.4746358 ,  0.47463425,  0.47463269,  0.47460914,
    #                   0.47460758,  0.47460603])

    work = np.array([0.6, 0.52, 0.51, 0.43, 0.40])
    #work = np.array([0.40, 0.52, 0.43, 0.51, 0.6])

    runtime = sim_runtime_of_work_package_heuristic(work, 3)
    print("runtime {}".format(runtime))

    num_wps = 1
    parallel_solvers_per_work_package = [3]

    work_packages = create_work_package(work, num_wps, SimRuntimeStrategy.asitis)
    work_packages_runtimes = sim_runtime_of_work_packages2(work_packages, parallel_solvers_per_work_package, SimRuntimeStrategy.asitis)
    print("a_t: {}".format(work_packages_runtimes[0]))
    print("a_t: {}".format(work_packages_runtimes[1]))
    print("a_t: {}".format(work))
    print("a_t: {}".format(np.array(work_packages_runtimes[2])))


    r1_start = time.time()
    min_t, max_t = find_optimal_and_worst_runtimes(work, num_wps, parallel_solvers_per_work_package)
    r1_end = time.time()
    print("r1:    {}".format(r1_end - r1_start))

    print("min_t: {}".format(min_t[0]))
    print("min_t: {}".format(min_t[1]))
    print("min_t: {}".format(np.array(min_t[2])))
    print("min_t: {}".format(np.array(min_t[3])))
    print("min_t: {}".format(min_t[4]))

    print("max_t: {}".format(max_t[0]))
    print("max_t: {}".format(max_t[1]))
    print("max_t: {}".format(np.array(max_t[2])))
    print("max_t: {}".format(np.array(max_t[3])))
    print("max_t: {}".format(max_t[4]))
    #
    # r2_start = time.time()
    # min_t, max_t = find_optimal_and_worst_runtimes1(work, 1, [2])
    # r2_end = time.time()
    #
    # print "r2: " + str(r2_end - r2_start)
    # print "min_t: " + str(min_t[0])
    # print "min_t: " + str(np.array(min_t[1]))
    # print "min_t: " + str(np.array(min_t[2]))
    #
    # print "max_t: " + str(max_t[0])
    # print "max_t: " + str(np.array(max_t[1]))
    # print "max_t: " + str(np.array(max_t[2]))

    r3_start = time.time()
    min_t, max_t = find_optimal_and_worst_runtimes3(work, num_wps, parallel_solvers_per_work_package)
    r3_end = time.time()

    print("r3:    {}".format(r3_end - r3_start))
    print("min_t: {}".format(min_t[0]))
    print("min_t: {}".format(min_t[1]))
    print("min_t: {}".format(np.array(min_t[2])))
    print("min_t: {}".format(np.array(min_t[3])))
    print("min_t: {}".format(min_t[4]))

    print("max_t: {}".format(max_t[0]))
    print("max_t: {}".format(max_t[1]))
    print("max_t: {}".format(np.array(max_t[2])))
    print("max_t: {}".format(np.array(max_t[3])))
    print("max_t: {}".format(max_t[4]))
