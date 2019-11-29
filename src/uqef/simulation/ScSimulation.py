"""
Stochastic collocation simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation


class ScSimulation(Simulation):
    """
    ScSimulation does a stochastic collocation simulation
    """

    def __init__(self, solver, q_order, p_order, rule="G", sparse_quadrature=False, regression=False, *args, **kwargs):
        Simulation.__init__(self, "sc", solver, *args, **kwargs)

        self.q_order = q_order
        self.p_order = p_order
        self.rule = rule
        self.sparse_quadrature = sparse_quadrature
        self.regression = regression

    def getSetup(self):
        return "%s using q_order=%d and p_order=%d %s" % (type(self).__name__, self.q_order, self.p_order, "with regression" if self.regression else "")

    def generateSimulationNodes(self, simulationNodes):
        nodes, weights, parameters= simulationNodes.generateNodesForSC(self.q_order, rule=self.rule, sparse=self.sparse_quadrature)
        nodes = nodes.T
        
        if parameters is not None:
            self.parameters = parameters.T
        else:
            self.parameters = nodes
        self.nodes = nodes
        self.weights = weights

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None):
        model_results = self.solver.results
        timesteps = self.solver.timesteps
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForSc(model_results, timesteps, simulationNodes, self.p_order,
                                       self.regression,
                                       solverTimes,
                                       self.solver.work_package_indexes, self.original_runtime_estimator)
        
        return statistics

