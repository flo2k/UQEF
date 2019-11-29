"""
A simple example for the usage of the UQEF with a test model.

@author: Florian Kuenzner
"""

# plotting
import matplotlib
matplotlib.use('Agg')

# numerical stuff
import chaospy as cp
import uqef

# instantiate UQsim
uqsim = uqef.UQsim()

# args:
uqsim.args.analyse_runtime = True

# initialise uncertain parameters:
if uqsim.is_master():
    uqsim.setup_nodes(["uncertain_param_1", "uncertain_param_2"])

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

# setup
uqsim.setup()

# start the simulation
uqsim.simulate()

# statistics:
uqsim.calc_statistics()
uqsim.print_statistics()
uqsim.plot_statistics()
uqsim.save_statistics()

# tear down
uqsim.tear_down()
