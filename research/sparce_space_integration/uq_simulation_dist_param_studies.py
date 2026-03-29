import chaospy as cp
import uqef

#                p1_1    p2_1       p1_2    p2_2
dist_params = [((0, 1), (-1, 1)), ((5, 2), (-3, 3))]

for dp1, dp2 in dist_params:
    # instantiate UQsim
    uqsim = uqef.UQsim()

    # args
    uqsim.args.sc_q_order = 3 # number of colloc. points for each param.
    uqsim.args.sc_p_order = 2 # highest order of orth. poly. of the gPCE

    # initialise uncertain parameters:
    if uqsim.is_master():
        uqsim.setup_nodes(["uncertain_param_1", "uncertain_param_2"])
        uqsim.simulationNodes.setDist("uncertain_param_1", cp.Normal(*dp1))
        uqsim.simulationNodes.setDist("uncertain_param_2", cp.Uniform(*dp2))

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
