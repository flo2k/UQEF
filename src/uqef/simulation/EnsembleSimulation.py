"""
EnsembleSimulation simulation class

@author: Ivana Jovanovic Buha
"""

from .Simulation import Simulation


class EnsembleSimulation(Simulation):
    """
    EnsembleSimulation does a monte carlo like simulation
    """

    def __init__(self, solver, *args, **kwargs):
        Simulation.__init__(self, "ensemble", solver, *args, **kwargs)

        self.numEvaluations = None
        self.parameters = None
        self.nodes = None

    def getSetup(self):
        return "%s running" % (type(self).__name__)

    def generateSimulationNodes(self, simulationNodes):
        nodes, parameters = simulationNodes.get_nodes_and_parameters()
        nodes = nodes.T

        self.numEvaluations = len(nodes)

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

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        model_results = self.solver.results
        timesteps = self.solver.timesteps
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForEnsemble(rawSamples=model_results,
                                       timesteps=timesteps,
                                       simulationNodes=simulationNodes,
                                       numEvaluations=self.numEvaluations,
                                       solverTimes=solverTimes,
                                       work_package_indexes=self.solver.work_package_indexes,
                                       original_runtime_estimator=self.original_runtime_estimator,
                                       *args, **kwargs)

        return statistics