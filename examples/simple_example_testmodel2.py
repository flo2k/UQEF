"""
A simple example for the usage of the UQEF with a test model.

@author: Florian Kuenzner
"""

# plotting
import matplotlib
matplotlib.use('Agg')

# parsing args
import argparse

# for parallel computing
import multiprocessing

# for message passing
from mpi4py import MPI

# numerical stuff
import chaospy as cp
import numpy as np
import uqef

# system stuff
import os


#####################################
### MPI infos:
#####################################

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#####################################
### parsing args:
#####################################
if rank == 0: print("parsing args...")
parser = argparse.ArgumentParser(description='Uncertainty Quantification simulation.')
parser.add_argument('--smoketest'             , action='store_true', default=False)

parser.add_argument('-or','--outputResultDir' , default=".")

parser.add_argument('--parallel'              , action='store_true', default=False)
parser.add_argument('--num_cores'             , type=int, default=multiprocessing.cpu_count())
parser.add_argument('--mpi'                   , action='store_true')
parser.add_argument('--mpi_method'            , default="new")  # new (MpiPoolSolver), old (MpiPoolSolverOld)
parser.add_argument('--mpi_combined_parallel' , action='store_true', default=False)

parser.add_argument('--model'                 , default="testmodel")

parser.add_argument('--chunksize'             , type=int, default=1)
parser.add_argument('--mpi_chunksize'         , type=int, default=1)

parser.add_argument('--uncertain'             , default='all')  # all, uncertain_param_1, uncertain_param_2
parser.add_argument('--uq_method'             , default="sc")  # sc, mc
parser.add_argument('--mc_numevaluations'     , type=int, default=27)
parser.add_argument('--sc_q_order'            , type=int, default=2)  # number of collocation points in each direction (Q)
parser.add_argument('--sc_p_order'            , type=int, default=1)  # number of terms in PCE (N)
parser.add_argument('--sc_sparse_quadrature'  , action='store_true', default=False)
parser.add_argument('--sc_quadrature_rule'    , default='g')

args = parser.parse_args()

#####################################
### smoke test:
#####################################

if args.smoketest == True:
    print("smoke test passed: exit!")
    exit(0)

#####################################
### parallelisation setup:
#####################################
# mpi
if args.mpi:
    mpi = True
else:
    mpi = False

# cores
num_cores = args.num_cores
if args.mpi and args.mpi_combined_parallel is False:
    num_cores = 1

if mpi == False or (mpi == True and rank == 0):
    print("set num cores to: {}".format(num_cores))

#####################################
### path settings:
#####################################
if rank == 0: print("path settings...")

model = args.model

if mpi == False or (mpi == True and rank == 0):
    if args.outputResultDir:
        outputResultDir = args.outputResultDir
    else:
        outputResultDir = os.getcwd()
    print("outputResultDir: {}".format(outputResultDir))

    #####################################
    ### initialise uncertain parameters:
    #####################################
    print("initialise uq parameters...")

    # parameter setup
    uncertain_param_1 = 3
    uncertain_param_1_std = 0.1
    uncertain_param_2 = 6
    uncertain_param_2_std = 0.1

    nodeNames = ["uncertain_param_1", "uncertain_param_2"]
    simulationNodes = uqef.simulation.Nodes(nodeNames)

    if args.uncertain == "all":
        simulationNodes.setDist("uncertain_param_1", cp.Normal(uncertain_param_1, uncertain_param_1_std))
        simulationNodes.setDist("uncertain_param_2", cp.Normal(uncertain_param_2, uncertain_param_2_std))
    elif args.uncertain == "uncertain_param_1":
        simulationNodes.setDist("uncertain_param_1", cp.Normal(uncertain_param_1, uncertain_param_1_std))
        simulationNodes.setValue("uncertain_param_2", uncertain_param_2)
    elif args.uncertain == "uncertain_param_2":
        simulationNodes.setValue("uncertain_param_1", uncertain_param_1)
        simulationNodes.setDist("uncertain_param_2", cp.Normal(uncertain_param_2, uncertain_param_2_std))

    print("model: {}".format(args.model))
    print("chunksize: {}".format(args.chunksize))
    print("nodes config: {}".format(args.uncertain))
    print(simulationNodes.printNodesSetup())


#####################################
### initialise model
#####################################

def modelGenerator():
    models = {
        "testmodel": (lambda: uqef.model.TestModel())
    }
    return models[model]()


#####################################
### initialise solver
#####################################

if mpi == True:
    if args.mpi_method == "new":
        solver = uqef.solver.MpiPoolSolver(modelGenerator, mpi_chunksize=args.mpi_chunksize,
                                           combinedParallel=args.mpi_combined_parallel, num_cores=num_cores)
    else:
        solver = uqef.solver.MpiSolverOld(modelGenerator, mpi_chunksize=args.mpi_chunksize,
                                              combinedParallel=args.mpi_combined_parallel, num_cores=num_cores)
elif args.parallel:
    solver = uqef.solver.ParallelSolver(modelGenerator, num_cores)
else:
    solver = uqef.solver.LinearSolver(modelGenerator)

if mpi == False or (mpi == True and rank == 0):
    print("solver-setup: {}".format(solver.getSetup()))

#####################################
### initialise simulation
#####################################
if mpi == False or (mpi == True and rank == 0):
    simulations = {
        "mc": (lambda: uqef.simulation.McSimulation(solver, args.mc_numevaluations))
       ,"sc": (lambda: uqef.simulation.ScSimulation(solver, args.sc_q_order, args.sc_p_order, args.sc_quadrature_rule, args.sc_sparse_quadrature))
    }
    simulation = simulations[args.uq_method]()

    print("simulation-setup: {}".format(simulation.getSetup()))


    #####################################
    ### initialise simulation:
    #####################################
    print("initialise simulation...")

    simulation.generateSimulationNodes(simulationNodes)
    print(simulationNodes.printNodes())

    # TODO: assert nodes

    #####################################
    ### start the simulation
    #####################################
    print("start the simulation...")
    solver.init()

    simulation.prepareSolver()

# do the solving => the propagation
solver.solve(chunksize=args.chunksize)

if mpi == False or (mpi == True and rank == 0):
    solver.tearDown() # stop the solver

    #####################################
    ### calculate statistics:
    #####################################
    print("calculate statistics...")

    statistics_ = {
        "testmodel": (lambda: simulation.calculateStatistics(uqef.stat.TestModelStatistics(), simulationNodes))
    }
    statistics = statistics_[model]()

    #####################################
    ### print statistics:
    #####################################
    print("print statistics...")
    print(statistics.printResults())

    #####################################
    ### generate plots
    #####################################
    print("generate plots...")
    fileName = simulation.name
    statistics.plotResults(fileName=fileName, directory=outputResultDir, display=False)

if mpi == True:
    print("rank: {} exit".format(rank))
