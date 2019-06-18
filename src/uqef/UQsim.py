

# parsing args
import argparse

# for parallel computing
import multiprocessing

# for message passing
from mpi4py import MPI

import uqef

# system stuff
import os

#####################################
### MPI infos:
#####################################

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

model = ""


def modelGenerator():
    models = {
        "testmodel": (lambda: uqef.model.TestModel())
    }
    return models[model]()


class UQsim(object):

    def __init__(self):
        self._init_parser()

        self.models = {
            "testmodel": (lambda: uqef.model.TestModel())
        }

    def __del__(self):
        if self.args.mpi is True:
            print("rank: {} exit".format(rank))

    def _init_parser(self):
        if rank == 0:
            print("parsing args...")

        self.parser = argparse.ArgumentParser(description='Uncertainty Quantification simulation.')

        self.parser.add_argument('--smoketest', action='store_true', default=False)

        self.parser.add_argument('-or', '--outputResultDir', default=".")

        self.parser.add_argument('--parallel', action='store_true', default=False)
        self.parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count())
        self.parser.add_argument('--mpi', action='store_true')
        self.parser.add_argument('--mpi_method', default="new")  # new (MpiPoolSolver), old (MpiPoolSolverOld)
        self.parser.add_argument('--mpi_combined_parallel', action='store_true', default=False)

        self.parser.add_argument('--model', default="testmodel")

        self.parser.add_argument('--chunksize', type=int, default=1)
        self.parser.add_argument('--mpi_chunksize', type=int, default=1)

        self.parser.add_argument('--uncertain', default='all')  # all, uncertain_param_1, uncertain_param_2
        self.parser.add_argument('--uq_method', default="sc")  # sc, mc
        self.parser.add_argument('--mc_numevaluations', type=int, default=27)
        self.parser.add_argument('--sc_q_order', type=int, default=2)  # number of collocation points in each direction (Q)
        self.parser.add_argument('--sc_p_order', type=int, default=1)  # number of terms in PCE (N)
        self.parser.add_argument('--sc_sparse_quadrature', action='store_true', default=False)
        self.parser.add_argument('--sc_quadrature_rule', default='g')

        #####################################
        ### smoke test:
        #####################################

        if self.args.smoketest is True:
            print("smoke test passed: exit!")
            exit(0)

    def is_master(self):
        return self.args.mpi is False or (self.args.mpi is True and rank == 0)

    def parse_args(self):
        self.args = self.parser.parse_args()

    def setup_path(self):
        #####################################
        ### path settings:
        #####################################
        if rank == 0:
            print("path settings...")

        if self.is_master():
            if not self.args.outputResultDir:
                self.args.outputResultDir = os.getcwd()
            print("outputResultDir: {}".format(self.args.outputResultDir))

    def setup_parallelisation(self):
        #####################################
        ### parallelisation setup:
        #####################################
        # cores
        if self.args.mpi and self.args.mpi_combined_parallel is False:
            self.args.num_cores = 1

        if self.is_master():
            print("set num cores to: {}".format(self.args.num_cores))

    def setup_nodes(self, nodeNames):
        self.simulationNodes = uqef.nodes.Nodes(nodeNames)

    def setup_model(self):
        global model
        model = self.args.model

    def initialise_solver(self):
        if self.args.mpi is True:
            if self.args.mpi_method == "new":
                self.solver = uqef.solver.MpiPoolSolver(modelGenerator, mpi_chunksize=self.args.mpi_chunksize,
                                                   combinedParallel=self.args.mpi_combined_parallel, num_cores=self.args.num_cores)
            else:
                self.solver = uqef.solver.MpiSolverOld(modelGenerator, mpi_chunksize=self.args.mpi_chunksize,
                                                  combinedParallel=self.args.mpi_combined_parallel, num_cores=self.args.num_cores)
        elif self.args.parallel:
            self.solver = uqef.solver.ParallelSolver(modelGenerator, self.args.num_cores)
        else:
            self.solver = uqef.solver.LinearSolver(modelGenerator)

        if self.args.mpi is False or (self.args.mpi is True and rank == 0):
            print("solver-setup: {}".format(self.solver.getSetup()))

    def initialise_simulation(self):
        #####################################
        ### initialise simulation
        #####################################
        if self.is_master():
            simulations = {
                "mc": (lambda: uqef.simulation.McSimulation(self.solver, self.args.mc_numevaluations))
               ,"sc": (lambda: uqef.simulation.ScSimulation(self.solver, self.args.sc_q_order, self.args.sc_p_order,
                                                            self.args.sc_quadrature_rule, self.args.sc_sparse_quadrature))
            }
            self.simulation = simulations[self.args.uq_method]()

            print("simulation-setup: {}".format(self.simulation.getSetup()))

            #####################################
            ### initialise simulation:
            #####################################
            print("initialise simulation...")

            self.simulation.generateSimulationNodes(self.simulationNodes)
            print(self.simulationNodes.printNodes())

            # TODO: assert nodes

    def simulate(self):
        if self.is_master():
            #####################################
            ### start the simulation
            #####################################
            print("start the simulation...")
            self.solver.init()

            self.simulation.prepareSolver()

        # do the solving => the propagation
        self.solver.solve(chunksize=self.args.chunksize)

        if self.is_master():
            self.solver.tearDown()  # stop the solver

    def calc_stat(self):
        #####################################
        ### calculate statistics:
        #####################################
        print("calculate statistics...")

        statistics_ = {
            "testmodel": (lambda: self.simulation.calculateStatistics(uqef.stat.TestModelStatistics(), self.simulationNodes))
        }
        self.statistics = statistics_[model]()

    def print_statistics(self):
        #####################################
        ### print statistics:
        #####################################
        print("print statistics...")
        print(self.statistics.printResults())

    def plot_statistics(self):
        #####################################
        ### generate plots
        #####################################
        print("generate plots...")
        fileName = self.simulation.name
        self.statistics.plotResults(fileName=fileName, directory=self.args.outputResultDir, display=False)
