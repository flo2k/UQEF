"""
A simple example for the usage of the UQEF with a test model.

@author: Florian Kuenzner
"""

# plotting
import matplotlib
matplotlib.use('Agg')

# numerical stuff
import uqef

# instantiate UQsim
uqsim = uqef.UQsim()

# change args locally
uqsim.args.analyse_runtime = True
uqsim.args.config_file = "config.json"

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
