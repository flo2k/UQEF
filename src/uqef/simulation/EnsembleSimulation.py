"""
EnsembleSimulation simulation class

@author: Ivana Jovanovic Buha
"""

from .Simulation import Simulation


class EnsembleSimulation(Simulation):
    """
    EnsembleSimulation does a Monte Carlo-like simulation
    """

    def __init__(self, solver, *args, **kwargs):
        Simulation.__init__(self, "ensemble", solver, *args, **kwargs)

        self.numEvaluations = None
        self.parameters = None
        self.nodes = None

    def getSetup(self):
        return "%s running" % (type(self).__name__)

    def generateSimulationNodes(self, simulationNodes, read_nodes_from_file=False, parameters_file_name=None,
                                parameters_setup_file_name=None):
        nodes, parameters = simulationNodes.get_nodes_and_parameters()
        self.numEvaluations = len(nodes.T)
        self.parameters = parameters.T
        self.nodes = nodes.T

    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        timesteps = self.solver.timesteps()
        statistics.prepare(rawSamples=self.solver.results,
                           timesteps=timesteps,
                           solverTimes=self.solver.solverTimes,
                           work_package_indexes=self.solver.work_package_indexes)
        statistics.prepareForEnsembleStatistics(
            simulationNodes, self.numEvaluations, *args, **kwargs)

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        model_results = self.solver.results
        timesteps = self.solver.timesteps
        solverTimes = self.solver.solverTimes
        self.original_runtime_estimator = original_runtime_estimator

        statistics.calcStatisticsForEnsemble(
            rawSamples=model_results,
            timesteps=timesteps,
            simulationNodes=simulationNodes,
            numEvaluations=self.numEvaluations,
            solverTimes=solverTimes,
            work_package_indexes=self.solver.work_package_indexes,
            original_runtime_estimator=self.original_runtime_estimator,
            *args, **kwargs
        )

        return statistics