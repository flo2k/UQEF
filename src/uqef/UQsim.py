# parsing args
import argparse

# for parallel computing
import multiprocessing

# for message passing
from mpi4py import MPI
import mpi4py

import chaospy as cp

import uqef

# time measure
import time
import datetime

# system stuff
import os
import sys
import inspect

import json
import dill
import pickle

#####################################
### MPI infos:
#####################################

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
version = MPI.Get_library_version()
version2 = MPI.Get_version()

if rank == 0: print("MPI version: {}".format(version))
if rank == 0: print("MPI2 version: {}".format(version2))
if rank == 0: print("MPI3 version: {}".format(MPI.VERSION))
if rank == 0: print("mpi4py version: {}".format(mpi4py.__version__))

print("rank {}: starttime: {}".format(rank, datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')))


def model_generator():
    return model_generator.model()


class UQsim(object):

    def __init__(self):
        self._init_parser()

        self.args = None
        self.configuration_object = None
        self.parse_args()

        if self.args.uqsim_restore_from_file:
            self.restore_from_file()
            self.parse_args()  # reparse args to overwrite old arguments
            self.__restored = True
        else:
            self.models = {
                "testmodel": (lambda: uqef.model.TestModel())
            }

            self.statistics = {
                "testmodel": (lambda: uqef.stat.TestModelStatistics()),
                "runtime": (lambda: uqef.stat.RuntimeStatistics())
            }

            self.simulationNodes = None
            self.simulation = None
            self.runtime_estimator = None
            self.solver = None
            self.statistic = None
            self.runtime_statistic = None
            self.__restored = False

    def __del__(self):
        if self.args.mpi is True:
            print("rank: {} exit".format(rank))

        sys.stdout.flush()

    def _init_parser(self):
        if rank == 0:
            print("parsing args...")

        self.parser = argparse.ArgumentParser(description='Uncertainty Quantification simulation.')

        # Smoketest: test run of script to verify environment
        self.parser.add_argument('--smoketest'                 , action='store_true', default=False)

        # UQsim load/restore
        self.parser.add_argument('--uqsim_file'                , default="uqsim.saved")
        self.parser.add_argument('--uqsim_store_to_file'       , action='store_true', default=False)
        self.parser.add_argument('--uqsim_restore_from_file'   , action='store_true', default=False)
        self.parser.add_argument('--disable_calc_statistics'   , action='store_true', default=False)
        self.parser.add_argument('--disable_recalc_statistics' , action='store_true', default=False)
        self.parser.add_argument('--disable_statistics'        , action='store_true', default=False)

        # Model and result directories
        self.parser.add_argument('-im', '--inputModelDir'      , default=".")
        self.parser.add_argument('-om', '--outputModelDir'     , default=".")
        self.parser.add_argument('-or', '--outputResultDir'    , default=".")
        self.parser.add_argument('-src', '--sourceDir'          , default=".")

        # Model settings
        self.parser.add_argument('--model'                     , default="testmodel")
        self.parser.add_argument('--model_variant'             , type=int, default=1)
        self.parser.add_argument('--simulate_wait'             , action='store_true', default=False)

        # UQ method and uncertain parameter settings
        self.parser.add_argument('--uncertain'                 , default='all')
        self.parser.add_argument('--uq_method'                 , default="sc")  # sc, mc, saltelli, ensemble
        self.parser.add_argument('--regression'                , action='store_true', default=False)
        self.parser.add_argument('--mc_numevaluations'         , type=int, default=27)
        self.parser.add_argument('--sc_q_order'                , type=int, default=2)  # number of collocation points in each direction (Q)
        self.parser.add_argument('--sc_p_order'                , type=int, default=1)  # number of terms in PCE (N)
        self.parser.add_argument('--sc_sparse_quadrature'      , action='store_true', default=False)
        self.parser.add_argument('--sc_quadrature_rule'        , default='G')
        self.parser.add_argument('--sc_poly_normed'            , action='store_true', default=False)
        self.parser.add_argument('--sc_poly_rule'              , default="three_terms_recurrence") # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
        self.parser.add_argument('--cross_truncation'          , type=float, default=1.0)
        self.parser.add_argument('--regression_model_type'     , type=str, default=None)  # None | "OLS" | "LARS"
        self.parser.add_argument('--sampling_rule'             , default='random')  # "sobol" | "latin_hypercube" | "halton"  | "hammersley"
        self.parser.add_argument('--transformToStandardDist'   , action='store_true', default=False)
        self.parser.add_argument('--sampleFromStandardDist'    , action='store_true', default=False)
        self.parser.add_argument('--config_file')
        self.parser.add_argument('--read_nodes_from_file'      , action='store_true', default=False)
        self.parser.add_argument('--parameters_file')
        self.parser.add_argument('--parameters_setup_file')

        # Solver settings
        self.parser.add_argument('--parallel'                  , action='store_true', default=False)
        self.parser.add_argument('--num_cores'                 , type=int, default=multiprocessing.cpu_count())
        self.parser.add_argument('--mpi'                       , action='store_true')
        self.parser.add_argument('--mpi_method'                , default="MpiPoolSolver")  # MpiPoolSolver: DWP, MpiSolver: SWP, SWPT
        self.parser.add_argument('--mpi_combined_parallel'     , action='store_true', default=False)

        # Statistics settings

        self.parser.add_argument('--instantly_save_results_for_each_time_step'       , action='store_true', default=False)
        self.parser.add_argument('--parallel_statistics'       , action='store_true', default=False)
        self.parser.add_argument('--compute_Sobol_t'           , action='store_true', default=False)
        self.parser.add_argument('--compute_Sobol_m'           , action='store_true', default=False)
        self.parser.add_argument('--compute_Sobol_m2'          , action='store_true', default=False)
        self.parser.add_argument('--save_all_simulations'      , action='store_true', default=False)  # This might be a lot of data
        self.parser.add_argument('--store_qoi_data_in_stat_dict'      , action='store_true', default=False)
        self.parser.add_argument('--store_gpce_surrogate_in_stat_dict'      , action='store_true', default=False)  # Only relevant for sc mode when the gPCE surrogate is produced
        self.parser.add_argument('--collect_and_save_state_data'      , action='store_true', default=False)  # Only relevant for models which produce some state data as well

        # Chunk parameters
        self.parser.add_argument('--chunksize'                 , type=int, default=1)
        self.parser.add_argument('--mpi_chunksize'             , type=int, default=1)

        # Runtime analysis and optimisation parameters
        self.parser.add_argument('--analyse_runtime'           , action='store_true', default=False)
        self.parser.add_argument('--opt_runtime'               , action='store_true', default=False)
        self.parser.add_argument('--opt_runtime_gpce_Dir'      , default=".")
        self.parser.add_argument('--opt_algorithm'             , default="LPT")             # FCFS LPT SPT MULTIFIT
        self.parser.add_argument('--opt_strategy'              , default="DYNAMIC")         # FIXED_ALTERNATE FIXED_LINEAR DYNAMIC

    def is_master(self):
        return self.args.mpi is False or (self.args.mpi is True and rank == 0)

    def get_size(self):
        return size

    def is_restored(self):
        return self.__restored

    def parse_args(self):
        self.args = self.parser.parse_args()

        self.setup_configuration_object()

        # smoke test
        if self.args.smoketest is True:
            print("smoke test passed: exit!")
            exit(0)

        if self.is_master():
            print("rank: {} is master!".format(rank))

    def setup(self):
        if not self.is_restored() and self.args.uqsim_restore_from_file is True:  # for locally configured restore
            self.restore_from_file()
            self.__restored = True
        if not self.is_restored():
            self.setup_configuration_object()
            self.setup_nodes_via_config_file_or_parameters_file()
            self.setup_path()
            self.setup_model()
            self.setup_parallelisation()
            self.setup_solver()
            self.setup_simulation()
            self.setup_runtime_estimator()

    def setup_path(self):
        if rank == 0:
            print("path settings...")

        if self.is_master():
            if not self.args.outputResultDir:
                self.args.outputResultDir = os.getcwd()
            print("outputResultDir: {}".format(self.args.outputResultDir))

    def setup_parallelisation(self):
        # cores
        if self.args.mpi is True and self.args.mpi_combined_parallel is False:
            self.args.num_cores = 1

        if self.is_master():
            print("set num cores to: {}".format(self.args.num_cores))

    def setup_configuration_object(self):
        if self.configuration_object is None and self.args.config_file:
            with open(self.args.config_file) as f:
                if self.is_master():
                    print("Loading config_file from {}".format(self.args.config_file))
                self.configuration_object = json.load(f)
                if self.is_master():
                    print(self.configuration_object)

    def setup_nodes(self, node_names):
        self.simulationNodes = uqef.nodes.Nodes(node_names)

    def setup_nodes_via_config_file_or_parameters_file(self):
        if self.is_master() and self.configuration_object is not None:
            print("Config nodes via config file: {}".format(self.args.config_file))

            # node names
            node_names = []
            for parameter_config in self.configuration_object["parameters"]:
                node_names.append(parameter_config["name"])
            self.setup_nodes(node_names)

            if self.args.uq_method == "ensemble" and self.args.read_nodes_from_file and self.args.parameters_file:
                # reading values of the nodes form a file
                print("Reading nodes values from parameters file {}".format(self.args.parameters_file))
                self.simulationNodes.generateNodesFromListOfValues(
                    read_nodes_from_file=self.args.read_nodes_from_file,
                    parameters_file_name=self.args.parameters_file)
            else:
                # branch for all other uq_methods ('sc', 'mc', 'saltelli')
                # and 'ensemble' when parameters_file is not specified
                if self.args.sampleFromStandardDist:
                    self.simulationNodes.setTransformation()

                for parameter_config in self.configuration_object["parameters"]:
                    if self.args.uq_method == "ensemble":
                        if "values_list" in parameter_config:
                            self.simulationNodes.setValue(parameter_config["name"], parameter_config["values_list"])
                        elif "default" in parameter_config:
                            self.simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
                        else:
                            raise Exception(f"Error in UQSim.setup_nodes_via_config_file_or_parameters_file() : "
                                            f" an ensemble simulation should be run, "
                                            f"but values_list or default entries for parameter values are missing")
                    elif parameter_config["distribution"] == "None":
                        # take default value(s) from config file
                        if "values_list" in parameter_config:
                            self.simulationNodes.setValue(parameter_config["name"], parameter_config["values_list"])
                        elif 'default' in parameter_config:
                            self.simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
                        else:
                            raise Exception(f"Error in UQSim.setup_nodes_via_config_file_or_parameters_file() : "
                                            f" distribution of a parameter is None, "
                                            f"but values_list or default entries are missing")
                    else:
                        # node values and distributions -> automatically maps dists and their parameters by reflection mechanisms
                        cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
                        dist_parameters_values = []
                        for p in cp_dist_signature.parameters:
                            dist_parameters_values.append(parameter_config[p])

                        self.simulationNodes.setDist(parameter_config["name"],
                                                     getattr(cp, parameter_config["distribution"])(
                                                         *dist_parameters_values))

                        if self.args.sampleFromStandardDist:
                            # for numerical stability work with nodes from 'standard' distributions,
                            # and use parameters for forcing the model
                            if parameter_config["distribution"] == "Uniform":
                                if (self.args.uq_method == "sc") or (self.args.uq_method == "mc" and self.args.regression):  # Gauss–Legendre quadrature
                                    self.simulationNodes.setStandardDist(parameter_config["name"],
                                                                         getattr(cp, parameter_config["distribution"])(
                                                                             lower=-1, upper=1
                                                                         ))
                                else:
                                    self.simulationNodes.setStandardDist(parameter_config["name"],
                                                                         getattr(cp, parameter_config["distribution"])(
                                                                             lower=0, upper=1
                                                                         ))
                            else:
                                self.simulationNodes.setStandardDist(parameter_config["name"],
                                                                     getattr(cp, parameter_config["distribution"])())

                if self.args.uq_method == "ensemble":
                    # in case of an ensemble method, when parameters_file is not specified,
                    # take a cross product of values_list of all parameters
                    self.simulationNodes.generateNodesFromListOfValues()

    def setup_model(self):
        model_generator.model = self.models[self.args.model]

    def setup_solver(self):
        if self.args.mpi is True:
            solvers = {
                "MpiPoolSolver": (lambda: uqef.solver.MpiPoolSolver(
                    model_generator, mpi_chunksize=self.args.mpi_chunksize,
                    combinedParallel=self.args.mpi_combined_parallel, num_cores=self.args.num_cores))
               ,"MpiSolver"    : (lambda: uqef.solver.MpiSolver(
                    model_generator, mpi_chunksize=self.args.mpi_chunksize,
                    combinedParallel=self.args.mpi_combined_parallel, num_cores=self.args.num_cores))
            }
            self.solver = solvers[self.args.mpi_method]()
        elif self.args.parallel:
            self.solver = uqef.solver.ParallelSolver(model_generator, self.args.num_cores)
        else:
            self.solver = uqef.solver.LinearSolver(model_generator)

        if self.args.mpi is False or (self.args.mpi is True and rank == 0):
            print("solver-setup: {}".format(self.solver.getSetup()))

    def setup_simulation(self):
        if self.is_master():
            simulations = {
                "mc"      : (lambda: uqef.simulation.McSimulation(
                    self.solver,
                    self.args.mc_numevaluations,
                    self.args.sc_p_order,
                    self.args.sampling_rule,
                    self.args.regression,
                    self.args.sc_poly_normed,
                    self.args.sc_poly_rule,
                    cross_truncation=self.args.cross_truncation,
                    regression_model_type=self.args.regression_model_type))
                , "sc"      : (lambda: uqef.simulation.ScSimulation(
                    self.solver,
                    self.args.sc_q_order,
                    self.args.sc_p_order,
                    self.args.sc_quadrature_rule,
                    self.args.sc_sparse_quadrature,
                    self.args.regression,
                    self.args.sc_poly_normed,
                    self.args.sc_poly_rule,
                    cross_truncation=self.args.cross_truncation,
                    regression_model_type=self.args.regression_model_type))
                , "saltelli": (lambda: uqef.simulation.SaltelliSimulation(
                    self.solver,
                    self.args.mc_numevaluations,
                    self.args.sc_p_order,
                    self.args.sampling_rule,
                    self.args.regression,
                    self.args.sc_poly_normed,
                    self.args.sc_poly_rule,
                    cross_truncation=self.args.cross_truncation))
                , "ensemble": (lambda: uqef.simulation.EnsembleSimulation(self.solver))
            }
            self.simulation = simulations[self.args.uq_method]()

            print("simulation-setup: {}".format(self.simulation.getSetup()))

            print("initialise simulation...")

            if self.args.read_nodes_from_file:
                self.simulation.generateSimulationNodes(
                    self.simulationNodes,
                    self.args.read_nodes_from_file,
                    self.args.parameters_file,
                    self.args.parameters_setup_file
                )
            else:
                self.simulation.generateSimulationNodes(self.simulationNodes)

            #self.simulation.saveParametersToFile(self.args.outputResultDir + "/simulation_parameters")

            print("")
            print("Nodes setup:")
            print(self.simulationNodes.printNodesSetup())
            # print("")
            # print("Nodes:")
            # print(self.simulationNodes.printNodes())

    def setup_runtime_estimator(self):
        if self.is_master():
            #####################################
            ### load solves if required:
            #####################################
            if self.args.opt_runtime:
                print("runtime optimisation enabled...")
                simulationFileName = self.args.opt_runtime_gpce_Dir + "/" + self.simulation.name + "_runtime" + ".stat"
                if os.path.isfile(simulationFileName):
                    print("Restore runtime estimator from: {}".format(simulationFileName))
                    runtime_statistics = uqef.stat.Statistics.restoreFromFile(simulationFileName)
                    self.runtime_estimator = runtime_statistics.runtime_estimator
                else:
                    print("No runtime estimator found in: {}".format(simulationFileName))
            else:
                print("runtime optimisation disabled...")

    def simulate(self):
        if self.is_restored() is False:
            if self.is_master():
                print("start the simulation...")
                self.solver.init()

                solver_time_start = time.time()

                # here is where solver gets self.simulation.parameters
                self.simulation.prepareSolver()

            algorithm = uqef.schedule.Algorithms[self.args.opt_algorithm]
            # algorithm = uqef.schedule.Algorithm.FCFS
            # algorithm = uqef.schedule.Algorithm.LPT
            # algorithm = uqef.schedule.Algorithm.SPT
            # algorithm = uqef.schedule.Algorithm.MULTIFIT

            strategy = uqef.schedule.Strategies[self.args.opt_strategy]
            # strategy  = uqef.schedule.Strategy.FIXED_ALTERNATE
            # strategy  = uqef.schedule.Strategy.FIXED_LINEAR
            # strategy  = uqef.schedule.Strategy.DYNAMIC

            if self.is_master():
                print("Opt algorithm: {}".format(
                    list(uqef.schedule.Algorithms.keys())[list(uqef.schedule.Algorithms.values()).index(algorithm)]))
                print("Opt strategy: {}".format(
                    list(uqef.schedule.Strategies.keys())[list(uqef.schedule.Strategies.values()).index(strategy)]))
                sys.stdout.flush()

            # do the solving => the propagation
            self.solver.solve(runtime_estimator=self.runtime_estimator, chunksize=self.args.chunksize,
                              algorithm=algorithm, strategy=strategy)

            if self.is_master():
                self.solver.tearDown()  # stop the solver

            if self.is_master():
                print("simulation done.")
                solver_time_end = time.time()
                solver_time = solver_time_end - solver_time_start
                print("solver time: {} sec".format(solver_time))
                sys.stdout.flush()

    def prepare_statistics(self, **kwargs):
        if self.args.disable_statistics is False:
            if self.args.mpi is True and self.args.parallel_statistics is True:
                # when parallel_statistics is set to True, eche process should have a Statistics object
                print("create statistics object for the [{}] model...".format(self.args.model))
                self.statistic = self.statistics[self.args.model]()
                if self.is_master():
                    self.simulation.prepareStatistic(self.statistic, self.simulationNodes)
                    print("preparing statistics [{}]...".format(type(self.statistic).__name__))
            else:
                if self.is_master():
                    print("create statistics object for the [{}] model".format(self.args.model))
                    self.statistic = self.statistics[self.args.model]()
                    self.simulation.prepareStatistic(self.statistic, self.simulationNodes)
                    print("preparing statistics [{}]...".format(type(self.statistic).__name__))

    def calc_statistics(self, **kwargs):
        if self.statistic is None:
            self.prepare_statistics()
        if self.args.disable_statistics is False and self.args.disable_calc_statistics \
                is False and self.args.disable_recalc_statistics is False:
            if self.args.mpi is True and self.args.parallel_statistics is True:
                if self.is_master():
                    print("calculate statistics [{}]...".format(type(self.statistic).__name__))
                calculateStatistics = {
                    "mc"       : (lambda: self.statistic.calcStatisticsForMcParallel(
                        chunksize=self.args.chunksize))
                    ,"sc"      : (lambda: self.statistic.calcStatisticsForScParallel(
                        chunksize=self.args.chunksize))
                    ,"saltelli": (lambda: self.statistic.calcStatisticsForMcSaltelliParallel(
                        chunksize=self.args.chunksize))
                    ,"ensemble": (lambda: self.statistic.calcStatisticsForEnsembleParallel(
                        chunksize=self.args.chunksize))
                }
                calculateStatistics[self.args.uq_method]()
            elif self.is_master():
                print("calculate statistics [{}]...".format(type(self.statistic).__name__))
                self.simulation.calculateStatistics(
                    self.statistic, self.simulationNodes, self.runtime_estimator, **kwargs)

            if self.is_master():
                if self.args.analyse_runtime is True and self.args.model == "runtime":
                    self.runtime_statistic = self.statistic
                elif self.args.analyse_runtime is True:
                    self.runtime_statistic = self.statistics["runtime"]()
                    print("calculate statistics [{}]...".format(type(self.runtime_statistic).__name__))
                    self.simulation.calculateStatistics(
                        self.runtime_statistic, self.simulationNodes, self.runtime_estimator, **kwargs)

    def print_statistics(self, **kwargs):
        if self.is_master() and self.args.disable_statistics is False:
            print("print statistics...")
            print(self.statistic.printResults(**kwargs))

            if self.args.analyse_runtime is True and self.args.model != "runtime":
                print(self.runtime_statistic.printResults(**kwargs))

    def plot_nodes(self, display=False, **kwargs):
        if self.is_master() and self.args.disable_statistics is False:
            print("generate node plots...")
            fileName = self.simulation.name
            self.simulationNodes.plotDists(fileName=fileName, directory=self.args.outputResultDir, display=display)

    def plot_statistics(self, display=False, **kwargs):
        if self.is_master() and self.args.disable_statistics is False:
            print("generate stat plots...")
            fileName = self.simulation.name
            self.statistic.plotResults(display=display, fileName=fileName, directory=self.args.outputResultDir,
                                       **kwargs)

            if self.args.analyse_runtime is True and self.args.model != "runtime":
                self.runtime_statistic.plotResults(display=display, fileName=fileName,
                                                   directory=self.args.outputResultDir, **kwargs)

    def plot_animations(self, timesteps, display=False, **kwargs):
        if self.is_master() and self.args.disable_statistics is False:
            print("generate stat animations...")
            fileName = self.simulation.name
            self.statistic.plotAnimation(
                timesteps, fileName=fileName, directory=self.args.outputResultDir, display=display, **kwargs)
            if self.args.analyse_runtime is True and self.args.model != "runtime":
                self.runtime_statistic.plotAnimation(
                    timesteps, fileName=fileName, directory=self.args.outputResultDir, display=display, **kwargs)

    def save_simulationNodes(self, fileName=None):
        if self.is_master():
            print("save simulation nodes...")
            if fileName is None:
                fileName = self.simulation.name
            self.simulationNodes.saveToFile(self.args.outputResultDir + "/" + fileName)

    def save_statistics(self, **kwargs):
        if self.is_master() and self.args.disable_statistics is False:
            print("save statistics...")

            fileName=None
            if 'fileName' in kwargs:
                fileName = kwargs.get('fileName')
            if fileName is None:
                fileName = self.simulation.name

            self.statistic.saveToFile(fileName=fileName, directory=self.args.outputResultDir, **kwargs)
            # self.statistics.saveAsNetCdf(timesteps=statistics.timesteps, fileName=fileName, directory=outputResultDir)
            # self.statistics.printCsv(fileName=fileName, directory=outputResultDir)
            # self.statistic.saveRuntimeData(fileName=fileName, directory=self.args.outputResultDir)

            if self.args.analyse_runtime is True:
                fileName = fileName + "_runtime"
                self.runtime_statistic.saveToFile(fileName=fileName, directory=self.args.outputResultDir, **kwargs)
                # statistics.saveAsNetCdf(timesteps=statistics.timesteps, fileName=fileName, directory=outputResultDir)
                #    statistics.printCsv(fileName=fileName, directory=outputResultDir)
                self.runtime_statistic.saveRuntimeData(fileName=fileName, directory=self.args.outputResultDir, **kwargs)

    def save_simulation(self, fileName=None):
        if self.is_master():
            print("save simulation...")
            if fileName is None:
                fileName = self.simulation.name
            self.simulation.saveToFile(self.args.outputResultDir + "/" + fileName)

    def save_simulation_parameters(self, fileName=None):
        if self.is_master():
            print("save simulation...")
            if fileName is None:
                fileName = "simulation_parameters"
            self.simulation.saveParametersToFile(self.args.outputResultDir + "/" + fileName)

    def get_simulation_parameters_shape(self):
        return self.simulation.get_simulation_parameters_shape()

    def tear_down(self):
        if self.is_master():
            self.store_to_file()

        print("rank {}: endtime: {}".format(rank, datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')))
        sys.stdout.flush()

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'rb') as f:
            return dill.load(f)

    def save_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            dill.dump(self, f)

    def store_to_file(self):
        if self.is_master() and self.args.uqsim_store_to_file:
            print("UQsim: save to file: {}".format(self.args.uqsim_file))
            self.save_to_file(self.args.uqsim_file)

    def restore_from_file(self):
        if self.args.uqsim_restore_from_file:
            print("UQsim: restore from file: {}".format(self.args.uqsim_restore_from_file))
            self.__dict__.update(self.load_from_file(self.args.uqsim_file).__dict__)
