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

from RuntimeSimModel import RuntimeSimModel

# instantiate UQsim
uqsim = uqef.UQsim()

# parsing args:
uqsim.parser.add_argument('--masterScenarioFile'        , default="Master.scenario")
uqsim.parser.add_argument('--runtimesim_time_mul_factor', type=float, default=1.0)
uqsim.parser.add_argument('--runtimesim_intrinsic_error', action='store_true', default=False)
uqsim.parse_args()

uqsim.args.analyse_runtime = True
uqsim.args.simulate_wait   = True
uqsim.args.num_cores = 2

uqsim.args.model = "runtime"
uqsim.args.model_variant = 1

# register model
uqsim.models.update({"runtime": (lambda: RuntimeSimModel(uqsim.args.outputModelDir, uqsim.args.inputModelDir, uqsim.args.masterScenarioFile,
                                                         uqsim.args.simulate_wait, uqsim.args.model_variant,
                                                         uqsim.args.runtimesim_time_mul_factor,
                                                         uqsim.args.runtimesim_intrinsic_error
                                                         ))})

# initialise uncertain parameters:
if uqsim.is_master():
    #                v1   v2    V   V = VADERE
    p_tab_p1_mean = [0.3,  0.0,  0.3]
    p_tab_p1_var  = [0.03, 0.03, 0.03]
    p_tab_p1_min  = [0.1, -2.5,  0.1]
    p_tab_p1_max  = [0.5,  2.5,  0.5]

    p_tab_p2_mean = [1.0,  0.0,  1.0]
    p_tab_p2_var  = [0.5,  0.5,  0.5]
    p_tab_p2_min  = [0.8, -2.0,  0.8]
    p_tab_p2_max  = [1.2,  2.0,  1.2]

    p_tab_p3_mean = [1.6, 10.0,  1.6]
    p_tab_p3_var  = [0.3,  0.3,  0.3]
    p_tab_p3_min  = [1.4,  5.0,  1.4]
    p_tab_p3_max  = [1.8, 15.0,  1.8]

    p_id = 0
    if uqsim.args.model == "vadere":
        p_id = 2
    elif uqsim.args.model_variant == 1:
        p_id = 0
    elif uqsim.model_variant == 2:
        p_id = 1

    #parameter setup
    familyPercentage     = p_tab_p1_mean[p_id] #0.3 # 0.3
    familyPercentage_var = p_tab_p1_var[p_id] #0.03
    familyPercentage_min = p_tab_p1_min[p_id] #0.1 #0.1
    familyPercentage_max = p_tab_p1_max[p_id] #0.5 #0.5
    childrenSpeed        = p_tab_p2_mean[p_id] #1.0
    childrenSpeed_var    = p_tab_p2_var[p_id] #0.5 #0.03
    childrenSpeed_min    = p_tab_p2_min[p_id] #0.8 #0.8
    childrenSpeed_max    = p_tab_p2_max[p_id] #1.2 #1.2
    parentSpeed          = p_tab_p3_mean[p_id] #1.6
    parentSpeed_var      = p_tab_p3_var[p_id] #0.3
    parentSpeed_min      = p_tab_p3_min[p_id] #1.4 #1.4
    parentSpeed_max      = p_tab_p3_max[p_id] #1.8 #1.8
    randomSeed           = 1
    simTimeStepLength    = 0.4 #0.4, 4.0

    nodeNames = ["familyPercentage", "childrenSpeed", "parentSpeed", "seed", "simTimeStepLength"]
    uqsim.setup_nodes(nodeNames)

    if uqsim.args.uncertain == "familyPercentage":
        uqsim.simulationNodes.setDist("familyPercentage", cp.Uniform(familyPercentage_min, familyPercentage_max))
        uqsim.simulationNodes.setValue("childrenSpeed", childrenSpeed)
        uqsim.simulationNodes.setValue("parentSpeed", parentSpeed)
    elif uqsim.args.uncertain == "childrenSpeed":
        uqsim.simulationNodes.setValue("familyPercentage", familyPercentage)
        uqsim.simulationNodes.setDist("childrenSpeed", cp.Uniform(childrenSpeed_min, childrenSpeed_max))
        uqsim.simulationNodes.setValue("parentSpeed", parentSpeed)
    elif uqsim.args.uncertain == "parentSpeed":
        uqsim.simulationNodes.setValue("familyPercentage", familyPercentage)
        uqsim.simulationNodes.setValue("childrenSpeed", childrenSpeed)
        uqsim.simulationNodes.setDist("parentSpeed", cp.Uniform(parentSpeed_min, parentSpeed_max))
    elif uqsim.args.uncertain == "all":
        uqsim.simulationNodes.setDist("familyPercentage", cp.Uniform(familyPercentage_min, familyPercentage_max))
        #uqsim.simulationNodes.setDist("familyPercentage", cp.Normal(familyPercentage, familyPercentage_var))
        uqsim.simulationNodes.setDist("childrenSpeed", cp.Uniform(childrenSpeed_min, childrenSpeed_max))
        #uqsim.simulationNodes.setDist("childrenSpeed", cp.Normal(childrenSpeed, childrenSpeed_var))
        uqsim.simulationNodes.setDist("parentSpeed", cp.Uniform(parentSpeed_min, parentSpeed_max))
    elif uqsim.args.uncertain == "x2":
        uqsim.simulationNodes.setValue("familyPercentage", familyPercentage)
        uqsim.simulationNodes.setDist("childrenSpeed", cp.Uniform(childrenSpeed_min, childrenSpeed_max))
        uqsim.simulationNodes.setDist("parentSpeed", cp.Uniform(parentSpeed_min, parentSpeed_max))

    # simulationNodes.setDist("familyPercentage", cp.Uniform(familyPercentage_min, familyPercentage_max))
    # simulationNodes.setValue("familyPercentage", familyPercentage)

    # simulationNodes.setDist("childrenSpeed", cp.Uniform(childrenSpeed_min, childrenSpeed_max))
    # simulationNodes.setValue("childrenSpeed", childrenSpeed)

    # simulationNodes.setDist("parentSpeed", cp.Uniform(parentSpeed_min, parentSpeed_max))
    # simulationNodes.setValue("parentSpeed", parentSpeed)

    uqsim.simulationNodes.setValue("seed", randomSeed)
    uqsim.simulationNodes.setValue("simTimeStepLength", simTimeStepLength)

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
