"""
Saltelli simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation
import numpy as np


class SaltelliSimulation(Simulation):
    """
    SaltelliSimulation performs MC Saltelli-like simulation
    """
    def __init__(self, solver, numEvaluations, p_order, rule="R", regression=False,
                 poly_normed=False, poly_rule="three_terms_recurrence", *args, **kwargs):
        Simulation.__init__(self, "saltelli", solver, *args, **kwargs)

        self.numEvaluations = numEvaluations
        self.p_order = p_order
        self.regression = regression
        self.rule = rule
        self.poly_normed = poly_normed
        self.poly_rule = poly_rule
        self.cross_truncation = kwargs.get("cross_truncation", 1.0)

        self.parameters = None
        self.nodes = None

    def getSetup(self):
        if self.parameters is not None:
            return "%s running %d evaluations" % (type(self).__name__, len(self.parameters))
        else:
            return "%s running %d evaluations" % (type(self).__name__, self.numEvaluations*2)

    def generateSimulationNodes(self, simulationNodes, read_nodes_from_file=False, parameters_file_name=None,
                                parameters_setup_file_name=None):
        nodes, parameters = simulationNodes.generateNodesForMC(
            numSamples=self.numEvaluations * 2, rule=self.rule,
            read_nodes_from_file=read_nodes_from_file,
            parameters_file_name=parameters_file_name,
            parameters_setup_file_name=parameters_setup_file_name
        )

        if parameters is not None:
            temp = parameters
        else:
            temp = nodes

        self.nodes = nodes.T

        d = simulationNodes.distNodes.shape[0]  # temp.shape[0]
        N = self.numEvaluations  # (temp.shape)[1] should be 2*N
        new_dim = N * (d + 2)
        print(f"MC & Saltelli INFO: D is {d}, N is {N}, total number of calculations will be {new_dim}")
        m1 = temp.T[:N].T  # m1.shape = (d,N)
        m2 = temp.T[N:].T  # m2.shape = (d,N)

        zeros = [0] * d
        ones = [1] * d
        matrix_A = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=zeros)
        matrix_B = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=ones)
        matrix_A_B = np.concatenate([self._get_matrix(matrix_A=m1, matrix_B=m2, indices=index) for index in np.eye(d, dtype=bool)], axis=1)
        self.parameters = np.concatenate([matrix_A, matrix_B, matrix_A_B], axis=1)
        self.parameters = self.parameters.T  # should be in Saltelli's case N*(d+2) x d

    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        timesteps = self.solver.timesteps()
        statistics.prepare(rawSamples=self.solver.results,
                           timesteps=timesteps,
                           solverTimes=self.solver.solverTimes,
                           work_package_indexes=self.solver.work_package_indexes)
        statistics.prepareForMcSaltelliStatistics(simulationNodes, self.numEvaluations, self.regression, self.p_order,
                                                  self.poly_normed, self.poly_rule, cross_truncation=self.cross_truncation)

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
            model_results = self.solver.results
            timesteps = self.solver.timesteps()
            solverTimes = self.solver.solverTimes
            self.original_runtime_estimator = original_runtime_estimator

            statistics.calcStatisticsForMcSaltelli(model_results, timesteps, simulationNodes,
                                                   self.numEvaluations, self.p_order,
                                                   self.regression, self.poly_normed, self.poly_rule, solverTimes,
                                                   self.solver.work_package_indexes, self.original_runtime_estimator, *args, **kwargs)
            return statistics  # TODO remove return?

    @staticmethod
    def _get_matrix(matrix_A, matrix_B, indices):
        """Retrieve Saltelli matrix.
        Input matrices should be of dimension dim x number_of_samples
        len(indices) should be equal to the dim

        Return: A_B matrix from Saltelli's 2010 paper
        """
        new = np.empty(matrix_A.shape)
        for idx in range(len(indices)):
            if indices[idx]:
                new[idx] = matrix_B[idx]
            else:
                new[idx] = matrix_A[idx]
        return new
