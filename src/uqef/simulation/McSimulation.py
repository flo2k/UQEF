"""
Monte Carlo simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation


class McSimulation(Simulation):
    """
    McSimulation does a Monte Carlo simulation
    """

    def __init__(self, solver, numEvaluations, p_order, rule="R", regression=False,
                 poly_normed=False, poly_rule="three_terms_recurrence", *args, **kwargs):
        Simulation.__init__(self, "mc", solver, *args, **kwargs)

        self.numEvaluations = numEvaluations
        self.p_order = p_order
        self.regression = regression
        self.rule = rule
        self.poly_normed = poly_normed
        self.poly_rule = poly_rule
        self.cross_truncation = kwargs.get("cross_truncation", 1.0)
        self.regression_model_type = kwargs.get("regression_model_type", None)

        self.parameters = None
        self.nodes = None

    def getSetup(self):
        return "%s running %d evaluations %s" % (type(self).__name__, self.numEvaluations, "with regression" if self.regression else "")

    def generateSimulationNodes(self, simulationNodes, read_nodes_from_file=False, parameters_file_name=None,
                                parameters_setup_file_name=None):
        nodes, parameters = simulationNodes.generateNodesForMC(
            self.numEvaluations, rule=self.rule,
            read_nodes_from_file=read_nodes_from_file,
            parameters_file_name=parameters_file_name,
            parameters_setup_file_name=parameters_setup_file_name
        )
        nodes = nodes.T

        if parameters is not None:
            self.parameters = parameters.T
        else:
            self.parameters = nodes
        self.nodes = nodes

    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        timesteps = self.solver.timesteps()
        statistics.prepare(rawSamples=self.solver.results,
                           timesteps=timesteps,
                           solverTimes=self.solver.solverTimes,
                           work_package_indexes=self.solver.work_package_indexes)
        statistics.prepareForMcStatistics(
            simulationNodes, self.numEvaluations, self.regression, self.p_order, 
            self.poly_normed, self.poly_rule, cross_truncation=self.cross_truncation, regression_model_type=self.regression_model_type, *args, **kwargs)

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        model_results = self.solver.results
        timesteps = self.solver.timesteps()
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForMc(model_results, timesteps, simulationNodes,
                                       self.numEvaluations, self.p_order,
                                       self.regression, solverTimes,
                                       self.solver.work_package_indexes, self.original_runtime_estimator, 
                                       poly_normed=self.poly_normed, poly_rule=self.poly_rule, cross_truncation=self.cross_truncation,
                                       *args, **kwargs)

        return statistics  # TODO remove return?