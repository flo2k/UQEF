"""
A simple example for the usage of the UQEF with a test model and to see the store and restore functionality.

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
#uqsim.args.uqsim_file = "uqsim.saved"
uqsim.args.uqsim_store_to_file = True
uqsim.args.uqsim_restore_from_file = True

# setup
uqsim.setup()

# turn statistics off here, after locally restored the UQsim instance
uqsim.args.disable_statistics = True

# start the simulation
uqsim.simulate()

# statistics:
uqsim.calc_statistics()
uqsim.print_statistics()
uqsim.plot_statistics()
uqsim.save_statistics()

uqsim = None
