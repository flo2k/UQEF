"""
Heuristic functions to schedule the work

@author: Florian Kuenzner
"""

import numpy as np
import more_itertools

from .defs import *


def _schedule_fixed(T_indexes, m, strategy):
    """
    Schedule the work fixed in the order of T_indexes, without taking into account how much work
    each work item has.

    This schedule gives all worker (almost) the same amount of work items (if len(T_indexes) % m == 0).

    :param T_indexes: are the indexes of the actual work
    :param m        : are the number of containers (processes or work packages)
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    P = [[] for _ in range(0, m)]  # reserve space for the containers

    if strategy == Strategy.FIXED_ALTERNATE:
        i = 0
        for j in T_indexes:
            P[i].append(j)
            i = (i + 1) % (m)
    elif strategy == Strategy.FIXED_LINEAR:
        P_it = list(more_itertools.divide(m, T_indexes))
        P = [list(p) for p in P_it]

    return P


def _schedule_dynamic(T, T_indexes, m):
    """
    Schedule the work dynamically in the order of T_indexes, with taking into account how much work
    each work item has.

    This schedule gives all worker (almost) the same amount of work.

    :param T        : are the work items (each item contains its work length)
    :param T_indexes: are the indexes of the actual work
    :param m        : are the number of containers (processes or work packages)
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    working = np.zeros(m)
    worker_end_time = np.full(m, np.nan)
    P = [[] for _ in range(0, m)]  # reserve space for the containers

    T_indexes_todo = list(T_indexes)
    current_time = 0.0

    while len(T_indexes_todo) != 0 or working.any():
        # start work
        for iw in range(0, len(working)):
            if working[iw] == 0 and len(T_indexes_todo) > 0:
                working[iw] = 1
                t_index = T_indexes_todo.pop(0)

                P[iw].append(t_index)
                worker_end_time[iw] = current_time + T[t_index]

        # stop work
        if working.any():
            worker_index = np.nanargmin(worker_end_time)
            current_time = worker_end_time[worker_index]

            # reset worker
            working[worker_index] = 0
            worker_end_time[worker_index] = np.nan

    return P


def _schedule(T, T_indexes, m, strategy):
    """
    Schedule the work dynamically or fixed, depending on the given strategy.

    :param T        : are the work items (each item contains its work length)
    :param T_indexes: are the indexes of the actual work
    :param m        : are the number of containers (processes or work packages)
    :param strategy : how the work should be scheduled (dynamic or fixed)
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    if strategy in [Strategy.FIXED_ALTERNATE, Strategy.FIXED_LINEAR]:
        P = _schedule_fixed(T_indexes, m, strategy)
    elif strategy == Strategy.DYNAMIC:
        P = _schedule_dynamic(T, T_indexes, m)

    return P

def FCFS(T, m, strategy):
    """
    FCFS - First come first served algorithm: a common approach to schedule the work

    :param T: is the actual work: T_i is one task (work) item
    :param m: are the number of containers (processes or work packages)
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    T_indexes = range(len(T))
    P = _schedule(T, T_indexes, m, strategy)

    return P


def FFD(T, C):
    """
    FFD - First fit decreasing algorithm: a heuristic for the bin-packing problem.

    The worst case ratio is:
     ** not known **

    :param T: is the actual work: T_i is one task (work) item
    :param C: is the bound (size of containers) to fit
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    T_decreasing = sorted(range(len(T)), key=lambda k: T[k])[::-1]  # sort decreasing (from higher to lower)

    P = [[]]
    for j in T_decreasing:
        # find fitting P[i] for T[j]
        k = -1
        m = len(P)

        # TODO: This algorithm could be optimized by not doing the sum in every loop - the sum should be cached.
        for i in range(0, m):
            if np.sum(T[P[i]]) + T[j] <= C:
                k = i
                break

        if k == -1:
            P.append([])

        # append to P
        P[k].append(j)

    return P


def MULTIFIT(T, m, k=100):
    """
    MULTIFIT - A heuristic for the Pm problem: Multiple processors with same size

    The worst case ratio is:
     Rm(MULTIFIT) <= 1.222

    :param T: is the actual work: T_i is one task (work) item
    :param m: are the number of containers (processes or work packages)
    :param k: is the number of iterations to find the near optimum schedule
    :return: P - the near optimal partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    Cl = max((1. / m) * np.asscalar(np.sum(T)), np.max(T))  # Cl = lower bound
    Cu = max((2. / m) * np.asscalar(np.sum(T)), np.max(T))  # Cu = upper bound

    # TODO: This algorithm could be optimized by checking if Cl and Cu has changed, to not
    #       do the k iterations even if it is not necessary.
    for i in range(k):
        C = (Cl + Cu) / 2.0
        if len(FFD(T, C)) <= m:
            Cu = C
        else:
            Cl = C

    P = FFD(T, Cu)

    # Because FFD returns only a P with the number of required containers we have to manually enhance it to size m
    while len(P) < m:
        P.append([])

    return P


def SPT(T, m, strategy):
    """
    SPT - Shortest processing time first: a common approach to schedule the work

    :param T: is the actual work: T_i is one task (work) item
    :param m: are the number of containers (processes or work packages)
    :return: P - the near optimal partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    T_increasing = sorted(range(len(T)), key=lambda k: T[k])  # sort increasing (from lower to higher)
    P = _schedule(T, T_increasing, m, strategy)

    return P


def LPT(T, m, strategy):
    """
    LPT - Longest processing time first: A heuristic for the Pm problem: Multiple processors with same size

    The worst case ratio is:
     Rm(LPT) = 4/3 - 1/(3m)

    :param T: is the actual work: T_i is one task (work) item
    :param m: are the number of containers (processes or work packages)
    :return: P - the near optimal partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    T_decreasing = sorted(range(len(T)), key=lambda k: T[k])[::-1]  # sort decreasing (from higher to lower)
    P = _schedule(T, T_decreasing, m, strategy)

    return P


def LPT_alternating(T, m):
    """
    LPT_alternating - Longest processing time first: A heuristic for the Pm problem: Multiple processors with same size
     -> Very similar to LPT, but it goes up and down when assigning the tasks to the containers (work packages)

    The worst case ratio is:
     ** not known **

    :param T: is the actual work: T_i is one task (work) item
    :param m: are the number of containers (processes or work packages)
    :return: P - the near optimal partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    T_decreasing = sorted(range(len(T)), key=lambda k: T[k])[::-1]  # sort decreasing (from higher to lower)

    P = [[] for _ in range(0, m)]

    class Direction(Enum):
        UP   = 1
        DOWN = 2

    mode = Direction.UP

    # TODO: Makes LPT_alternating also sense with Strategy::DYNAMIC?
    i = 0
    for j in T_decreasing:
        P[i].append(j)

        #index calc (up -> down -> up -> ...)
        if mode == Direction.UP:
            i += 1
            if i >= m:
                i -= 1
                mode = Direction.DOWN
        elif mode == Direction.DOWN:
            i -= 1
            if i < 0:
                i += 1
                mode = Direction.DOWN

    return P


# TODO: remove the "__main__" function and write some UnitTests for it!
if __name__ == "__main__":
    """
    Some small usage and test examples...
    """

    # work = np.array([0.6, 0.52, 0.51, 0.43, 0.40])
    #work = np.array([0.52, 0.6, 0.51, 0.43, 0.40])
    work = np.array([52, 60, 51, 43, 40])
    #work = np.array([1.2, 1.9, 6.7, 3.5, 5.1, 2.2, 4.1, 8.5, 1.5])

    m = 3 #number of processors
    k = 100 #number of iterations to find near optimum
    s = Strategy.DYNAMIC

    T_decreasing = np.asarray(sorted(range(len(work)), key=lambda k: work[k])[::-1])  # sort decreasing (from higher to lower)

    print("Work: {}".format(work))
    print("Total work length: {}".format(np.sum(work)))
    print("Sorted LPT indizes: {}".format(T_decreasing))
    print("Sorted LPT work: {}".format([work[i] for i in T_decreasing]))
    print("Processors: {}".format(m))
    print("Perfect work length: {}".format(np.sum(work)/m))
    print("")

    ###################
    print("FCFS:")
    P = FCFS(work, m, s)

    print("P: {}".format(P))
    for i in range(0, len(P)):
        print("P[{}]: {}".format(i, P[i]))
    print("R: {}".format([list(work[P[i]]) for i in range(len(P))]))

    runtime = [np.sum(work[p]) for p in P]
    print("P runtimes: {}".format(runtime))
    print("Max runtime: {}".format(np.max(runtime)))
    print("ratio: {}".format(np.max(runtime) / (np.sum(work) / m)))
    print("")

    ###################
    print("SPT:")
    P = SPT(work, m, s)

    print("P: {}".format(P))
    for i in range(0, len(P)):
        print("P[{}]: {}".format(i, P[i]))
    print("R: {}".format([list(work[P[i]]) for i in range(len(P))]))

    runtime = [np.sum(work[p]) for p in P]
    print("P runtimes: {}".format(runtime))
    print("Max runtime: {}".format(np.max(runtime)))
    print("ratio: {}".format(np.max(runtime) / (np.sum(work) / m)))
    print("")

    ###################
    print("LPT:")
    P = LPT(work, m, s)

    print("P: {}".format(P))
    for i in range(0, len(P)):
        print("P[{}]: {}".format(i, P[i]))
    print("R: {}".format([list(work[P[i]]) for i in range(len(P))]))

    runtime = [np.sum(work[p]) for p in P]
    print("P runtimes: {}".format(runtime))
    print("Max runtime: {}".format(np.max(runtime)))
    print("ratio: {}".format(np.max(runtime) / (np.sum(work) / m)))
    print("")

    ###################
    print("LPT_alternating:")
    P = LPT_alternating(work, m)

    print("P: {}".format(P))
    for i in range(0, len(P)):
        print("P[{}]: {}".format(i, P[i]))
    print("R: {}".format([list(work[P[i]]) for i in range(len(P))]))

    runtime = [np.sum(work[p]) for p in P]
    print("P runtimes: {}".format(runtime))
    print("Max runtime: {}".format(np.max(runtime)))
    print("ratio: {}".format(np.max(runtime) / (np.sum(work) / m)))
    print("")

    ###################
    print("MULTIFIT:")
    P = MULTIFIT(work, m, k)

    print("P: {}".format(P))
    for i in range(0, len(P)):
        print("P[{}]: {}".format(i, P[i]))
    print("R: {}".format([list(work[P[i]]) for i in range(len(P))]))

    runtime = [np.sum(work[p]) for p in P]
    print("P runtimes: {}".format(runtime))
    print("Max runtime: {}".format(np.max(runtime)))
    print("ratio: {}".format(np.max(runtime)/(np.sum(work) / m)))
    print("")



