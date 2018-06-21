"""
Monte Carlo simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation


class McSimulation(Simulation):
    """
    ScSimulation does a monte carlo simulation
    """

    def __init__(self, solver, numEvaluations, *args, **kwargs):
        Simulation.__init__(self, "mc", solver, *args, **kwargs)

        self.numEvaluations = numEvaluations

    def getSetup(self):
        return "%s running %d evaluations" % (type(self).__name__, self.numEvaluations)

    def generateSimulationNodes(self, simulationNodes):
        nodes = simulationNodes.generateNodesForMC(self.numEvaluations)
        nodes = nodes.T
        self.parameters = nodes

    def calculateStatistics(self, statistics, simulationNodes):
        model_results = self.solver.results
        timesteps = self.solver.timesteps
        statistics.calcStatisticsForMc(model_results, timesteps, self.numEvaluations)
        
        return statistics