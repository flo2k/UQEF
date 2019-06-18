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


uqsim = uqef.UQsim()

uqsim.parse_args()
uqsim.setup_path()
uqsim.setup_parallelisation()
uqsim.setup_nodes(["uncertain_param_1", "uncertain_param_2"])

uqsim.simulationNodes

if uqsim.is_master():
    # parameter setup
    uncertain_param_1 = 3
    uncertain_param_1_std = 0.1
    uncertain_param_2 = 6
    uncertain_param_2_std = 0.1

    if uqsim.args.uncertain == "all":
        uqsim.simulationNodes.setDist("uncertain_param_1", cp.Normal(uncertain_param_1, uncertain_param_1_std))
        uqsim.simulationNodes.setDist("uncertain_param_2", cp.Normal(uncertain_param_2, uncertain_param_2_std))
    elif uqsim.args.uncertain == "uncertain_param_1":
        uqsim.simulationNodes.setDist("uncertain_param_1", cp.Normal(uncertain_param_1, uncertain_param_1_std))
        uqsim.simulationNodes.setValue("uncertain_param_2", uncertain_param_2)
    elif uqsim.args.uncertain == "uncertain_param_2":
        uqsim.simulationNodes.setValue("uncertain_param_1", uncertain_param_1)
        uqsim.simulationNodes.setDist("uncertain_param_2", cp.Normal(uncertain_param_2, uncertain_param_2_std))


uqsim.setup_model()

uqsim.initialise_solver()
uqsim.initialise_simulation()
uqsim.simulate()

uqsim.calc_stat()
uqsim.print_statistics()
uqsim.plot_statistics()