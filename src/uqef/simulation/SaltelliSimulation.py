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
        """
        Generate simulation nodes for Saltelli simulation
        Important note about dimensionality of different numpy arrays:
        - simulationNodes.distNodes.shape[0] should present a stochastic dimensionionality
        - simulationNodes.distNodes is of shape (stochstic_dim, numEvaluations*2)
        - simulationNodes.nodes is of shape (dim, numEvaluations*2)
        - simulationNodes.parameters is of shape (dim, numEvaluations*2)
        - self.parameters is of shape (numEvaluations*( stochastic dimensionionality + 2), dim)
        """
        nodes, parameters = simulationNodes.generateNodesForMC(
            numSamples=self.numEvaluations * 2, rule=self.rule,
            read_nodes_from_file=read_nodes_from_file,
            parameters_file_name=parameters_file_name,
            parameters_setup_file_name=parameters_setup_file_name
        )

        if parameters is not None:
            original_set_of_parameters = parameters
        else:
            original_set_of_parameters = nodes

        self.nodes = nodes.T

        dim = simulationNodes.distNodes.shape[0]  # this should present a stochastic dimensionionality
        N = self.numEvaluations  # (original_set_of_parameters.shape)[1] should be 2*N
        total_number_model_evaluations = N * (dim + 2)
        print(f"MC (Saltelli) INFO: D is {dim}, N is {N}, total number of calculations will be {total_number_model_evaluations}")
        m1 = original_set_of_parameters.T[:N].T  # m1.shape = (dim,N)
        m2 = original_set_of_parameters.T[N:].T  # m2.shape = (dim,N)

        zeros = [0] * dim
        ones = [1] * dim
        matrix_A = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=zeros)
        matrix_B = self._get_matrix(matrix_A=m1, matrix_B=m2, indices=ones)
        matrix_A_B = np.concatenate([self._get_matrix(matrix_A=m1, matrix_B=m2, indices=index) for index in np.eye(dim, dtype=bool)], axis=1)
        self.parameters = np.concatenate([matrix_A, matrix_B, matrix_A_B], axis=1)
        self.parameters = self.parameters.T  # should be in Saltelli's case N*(dim+2) x dim

    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        timesteps = self.solver.timesteps()
        statistics.prepare(rawSamples=self.solver.results,
                           timesteps=timesteps,
                           solverTimes=self.solver.solverTimes,
                           work_package_indexes=self.solver.work_package_indexes)
        statistics.prepareForMcSaltelliStatistics(simulationNodes, self.numEvaluations, self.regression, self.p_order,
                                                  self.poly_normed, self.poly_rule, cross_truncation=self.cross_truncation, *args, **kwargs)

    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
            model_results = self.solver.results
            timesteps = self.solver.timesteps()
            solverTimes = self.solver.solverTimes
            self.original_runtime_estimator = original_runtime_estimator

            statistics.calcStatisticsForMcSaltelli(model_results, timesteps, simulationNodes,
                                                   self.numEvaluations, self.p_order,
                                                   self.regression, solverTimes,
                                                   self.solver.work_package_indexes, self.original_runtime_estimator, 
                                                   poly_normed=self.poly_normed, poly_rule=self.poly_rule, cross_truncation=self.cross_truncation,
                                                   *args, **kwargs)
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
