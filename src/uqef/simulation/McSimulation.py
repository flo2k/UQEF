"""
Monte Carlo simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation


class McSimulation(Simulation):
    """
    ScSimulation does a monte carlo simulation
    """

    def __init__(self, solver, numEvaluations, p_order, regression=False, *args, **kwargs):
        Simulation.__init__(self, "mc", solver, *args, **kwargs)

        self.numEvaluations = numEvaluations
        self.p_order = p_order
        self.regression = regression

    def getSetup(self):
        return "%s running %d evaluations %s" % (type(self).__name__, self.numEvaluations, "with regression" if self.regression else "")

    def generateSimulationNodes(self, simulationNodes):
        nodes = simulationNodes.generateNodesForMC(self.numEvaluations)
        nodes = nodes.T
        self.parameters = nodes

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None):
        model_results = self.solver.results
        timesteps = self.solver.timesteps
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForMc(model_results, timesteps, simulationNodes,
                                       self.numEvaluations, self.p_order,
                                       self.regression,
                                       solverTimes,
                                       self.solver.work_package_indexes, self.original_runtime_estimator)
        
        return statistics