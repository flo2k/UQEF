"""
Stochastic collocation simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation


class ScSimulation(Simulation):
    """
    ScSimulation does a pseudo spectra projection scheme (regression is set to False)
     or stochastic collocation simulation (regression is set to True)
    """

    def __init__(self, solver, q_order, p_order, rule="G", sparse_quadrature=False, regression=False,
                 poly_normed=False, poly_rule="three_terms_recurrence", *args, **kwargs):
        Simulation.__init__(self, "sc", solver, *args, **kwargs)

        self.q_order = q_order
        self.p_order = p_order
        self.rule = rule
        self.sparse_quadrature = sparse_quadrature
        self.regression = regression
        self.poly_normed = poly_normed
        self.poly_rule = poly_rule
        self.cross_truncation = kwargs.get("cross_truncation", 1.0)
        self.regression_model_type = kwargs.get("regression_model_type", None)

        self.parameters = None
        self.nodes = None
        self.weights = None

    def getSetup(self):
        return "%s using q_order=%d and p_order=%d %s" % (type(self).__name__, self.q_order, self.p_order, "with regression" if self.regression else "")

    def generateSimulationNodes(self, simulationNodes, read_nodes_from_file=False, parameters_file_name=None,
                                parameters_setup_file_name=None):
        nodes, weights, parameters = simulationNodes.generateNodesForSC(
            self.q_order, rule=self.rule,
            sparse=self.sparse_quadrature,
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
        self.weights = weights

    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        timesteps = self.solver.timesteps()
        statistics.prepare(rawSamples=self.solver.results,
                           timesteps=timesteps,
                           solverTimes=self.solver.solverTimes,
                           work_package_indexes=self.solver.work_package_indexes)
        statistics.prepareForScStatistics(
            simulationNodes, self.p_order, self.poly_normed, self.poly_rule, self.regression, cross_truncation=self.cross_truncation, regression_model_type=self.regression_model_type, *args, **kwargs)

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        model_results = self.solver.results
        timesteps = self.solver.timesteps()
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForSc(model_results, timesteps, simulationNodes, self.p_order,
                                       self.regression,
                                       self.poly_normed, self.poly_rule,
                                       solverTimes,
                                       self.solver.work_package_indexes, self.original_runtime_estimator,
                                       cross_truncation=self.cross_truncation,
                                       *args, **kwargs)
        
        return statistics  # TODO remove return?

