"""
Saltelli simulation class

@author: Florian Kuenzner
"""

from .Simulation import Simulation
import numpy as np


class SaltelliSimulation(Simulation):
    """
    SaltelliSimulation does a saltelli simulation
    """

    def __init__(self, solver, numEvaluations, p_order, regression=False, *args, **kwargs):
        Simulation.__init__(self, "mc", solver, *args, **kwargs)

        self.numEvaluations = numEvaluations
        self.p_order = p_order
        self.regression = regression

    def getSetup(self):
        return "%s running %d evaluations" % (type(self).__name__, self.numEvaluations*2)

    def generateSimulationNodes(self, simulationNodes):
        nodes, parameters = simulationNodes.generateNodesForMC(self.numEvaluations*2)

        if parameters is not None:
            temp = parameters
        else:
            temp = nodes
        self.nodes = nodes

        d = temp.shape[0]
        N = self.numEvaluations  # (temp.shape)[1] should be 2*N
        new_dim = N * (d + 2)
        print("MC & Saltelli INFO: D is %d, N is %d, total number of calcuations will be %d" % (d, N, new_dim))
        m1 = temp.T[:N].T  # m1.shape = (d,N)
        m2 = temp.T[N:].T  # m2.shape = (d,N)
        print("MC & Saltelli INFO:m1 shape: {}".format(m1.shape))
        print("MC & Saltelli INFO:m2 shape: {}".format(m2.shape))

        zeros = [0] * d
        ones = [1] * d
        matrix_A = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=zeros)
        matrix_B = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=ones)
        matrix_A_B = np.concatenate([self._get_matrix(matrix_A=m1, matrix_B=m2, indices=index) for index in np.eye(d, dtype=bool)], axis=1)
        self.parameters = np.concatenate([matrix_A, matrix_B, matrix_A_B], axis=1)

        self.parameters = self.parameters.T  # should be in Saltelli's case N*(d+2) x d
        print("MC & Saltelli INFO: simulation.parameters shape ")
        print(self.parameters.shape)


    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None):
            model_results = self.solver.results
            timesteps = self.solver.timesteps
            solverTimes = self.solver.solverTimes
            self.original_runtime_estimator = original_runtime_estimator

            statistics.calcStatisticsForSaltelli(model_results, timesteps, simulationNodes,
                                                 self.numEvaluations, self.p_order,
                                                 self.regression,
                                                 solverTimes,
                                                 self.solver.work_package_indexes, self.original_runtime_estimator)

            return statistics

    @staticmethod
    def _get_matrix(matrix_A, matrix_B, indices):
        """Retrieve Saltelli matrix.
        Input matrices shoul be of dimension dim x number_of_samples
        len(indices) shoul be equal to the dim

        Return: A_B matrix from Saltelli 2010 paper
        """
        new = np.empty(matrix_A.shape)
        for idx in range(len(indices)):
            if indices[idx]:
                new[idx] = matrix_B[idx]
            else:
                new[idx] = matrix_A[idx]
        return new
