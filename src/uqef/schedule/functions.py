"""
Helper functions to generate and work with work packages

@author: Florian Kuenzner
"""

from .heuristics import *

# TODO: is this table still valid?
# valid/reasonable combinations
# Type         | Strategy                               | Algorithm
# -------------|----------------------------------------|------------------------------------------
#              | FIXED_ALTERNATE  FIXED_LINEAR  DYNAMIC | FCFS       LPT        SPT        MULTIFIT
# -------------|----------------------------------------|------------------------------------------
# WORK_LIST    |                                X       | X          X          X          -
# WORK_LIST    | X                                      | X          X          X          -
# WORK_LIST    |                  X                     | X          X          X          -

# WORK_PACKAGE |                                X       | X          X          X          -
# WORK_PACKAGE | X                                      | X          X          X          X
# WORK_PACKAGE |                  X                     | X          X          X          -


def generate_work_package(work, num_worker, algorithm, strategy):
    """
    Generates a work package considering the num_worker, the algorithm, and the strategy

    :param work      : is the actual work that has to be proceed. Each element/item contains it estimated runtime.
    :param num_worker: are the number of workers (each worker has the same execution speed and resources)
    :param algorithm : is the algorithm to calculate the scheduling order
    :param strategy  : is the strategy how the schedule algorithm works
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    if algorithm == Algorithm.FCFS:
        P = FCFS(work, num_worker, strategy)
    elif algorithm == Algorithm.LPT:
        P = LPT(work, num_worker, strategy)
    elif algorithm == Algorithm.SPT:
        P = SPT(work, num_worker, strategy)
    elif algorithm == Algorithm.MULTIFIT:
        P = MULTIFIT(work, num_worker)

    return P


def simulate_work_schedule(work, num_worker, type, algorithm, strategy):
    """
    Simulates the work schedule by the given work

    :param work      : is the actual work that has to be proceed. Each element/item contains it estimated runtime.
    :param num_worker: are the number of workers (each worker has the same execution speed and resources)
    :param type      : is the type how the work should be scheduled
    :param algorithm : is the algorithm to calculate the scheduling order
    :param strategy  : is the strategy how the schedule algorithm works
    :return: P - the partitioned indices packages (work packages) (P contains only the indices, for T)
    """
    # TODO: Does this Type parameter make sense? Should I delete it?
    if type == Type.WORK_LIST:
        num_worker = 1  # set the number of workers to 1!!

    # simulate the work
    P = generate_work_package(work, num_worker, algorithm, strategy)

    return P


def generate_work_list_from_work_package(work, P):
    """
    Generates a sorted work list as in the order a scheduler would dynamically schedule the work as given in P

    :param work: is the actual work that has to be proceed. Each element/item contains it estimated runtime.
    :param P   : the partitioned indices packages (work packages) (P contains only the indices, for T)
    :return: L - is the sorted work list
    """
    L = []
    m = len(P)
    iPs = [0] * m

    working = np.zeros(m)
    worker_end_time = np.full(m, np.nan)
    current_time = 0

    while(len(L) != len(work)):
        # start work
        for iW in range(0, m):
            if working[iW] == 0 and iPs[iW] < len(P[iW]):
                working[iW] = 0
                work_index = P[iW][iPs[iW]]
                worker_end_time[iW] = current_time + work[work_index]
                iPs[iW] += 1

                L.append(work_index)

        # stop work
        if working.any():
            worker_index = np.nanargmin(worker_end_time)
            current_time = worker_end_time[worker_index]

            # reset worker
            working[worker_index] = 0
            worker_end_time[worker_index] = np.nan


    return L


def calc_amount_of_work(work):
    """
    Calculates the amount of work. It is the sum of all work, as it would be done by a single processing unit.

    :param work: are the work items (each item contains its work length)
    :return: the amount (sum) of all work (type: scalar value)
    """
    return sum(work)


def calc_runtimes_for_each_workpackage(work, P):
    """
    Calculates the runtime for each workpackage.

    :param work: are the work items (each item contains its work length)
    :param P   : the partitioned indices packages (work packages) (P contains only the indices, for work)
    :return: the runtime for each partition (work package) in P. (type list: The length of the runtime list = len(P))
    """
    return [sum([work[iip] for iip in ip]) for ip in P]


def calc_complete_runtime(work, P):
    """
    Calculates the complete runtime by taking into account the individual runtime of each work package.

    :param work: are the work items (each item contains its work length)
    :param P   : the partitioned indices packages (work packages) (P contains only the indices, for work)
    :return: the complete runtime (type: scalar value)
    """
    return max(calc_runtimes_for_each_workpackage(work, P))

# TODO: remove the "__main__" function and write some UnitTests for it!
if __name__ == "__main__":
    """
    Some small usage and test examples...
    """

    #work = [3, 4, 5, 1, 2]
    work = range(1, 100, 2)
    work = np.asarray(work)

    num_worker = 4
    strategy = Strategy.FIXED_ALTERNATE

    def print_results():
        print("indexes  : {}".format(P_indexes))
        print("work list: {}".format(generate_work_list_from_work_package(work, P_indexes)))
        print("work     : {}".format([[work[iiwp] for iiwp in iwp] for iwp in P_indexes]))
        print("runtimes : {}".format(calc_runtimes_for_each_workpackage(work, P_indexes)))
        print("all work : {}".format(calc_amount_of_work(work)))
        print("runtime  : {}".format(calc_complete_runtime(work, P_indexes)))
        print("")

    #WORK_LIST
    print("WORK_LIST:FCFS:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_LIST, Algorithm.FCFS, strategy)
    print_results()

    print("WORK_LIST:LPT:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_LIST, Algorithm.LPT, strategy)
    print_results()

    print("WORK_LIST:SPT:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_LIST, Algorithm.SPT, strategy)
    print_results()

    # WORK_PACKAGE
    print("WORK_PACKAGE:FCFS:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_PACKAGE, Algorithm.FCFS, strategy)
    print_results()

    print("WORK_PACKAGE:LPT:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_PACKAGE, Algorithm.LPT, strategy)
    print_results()

    print("WORK_PACKAGE:SPT:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_PACKAGE, Algorithm.SPT, strategy)
    print_results()

    print("WORK_PACKAGE:MULTIFIT:")
    P_indexes = simulate_work_schedule(work, num_worker, Type.WORK_PACKAGE, Algorithm.MULTIFIT, strategy)
    print_results()