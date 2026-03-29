import chaospy as cp
import uqef

#          q  p    q  p    q  p    q  p
orders = [(3, 1), (4, 1), (5, 2), (6, 2)]

for q_order, p_order in orders:
    # instantiate UQsim
    uqsim = uqef.UQsim()

    # args
    uqsim.args.sc_q_order = q_order # number of colloc. points for each param.
    uqsim.args.sc_p_order = p_order # highest order of orth. poly. of the gPCE

    # initialise uncertain parameters:
    if uqsim.is_master():
        uqsim.setup_nodes(["uncertain_param_1", "uncertain_param_2"])
        uqsim.simulationNodes.setDist("uncertain_param_1", cp.Normal(0, 1))
        uqsim.simulationNodes.setDist("uncertain_param_2", cp.Uniform(-1, 1))

    # ... setup, simulate, statistics, tear down

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
