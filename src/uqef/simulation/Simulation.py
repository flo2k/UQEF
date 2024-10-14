"""
Abstract base class for simulations like: Monte Carlo, or stochastic collocation

@author: Florian Kuenzner
"""

from abc import abstractmethod
from abc import ABCMeta

import dill
import numpy as np


class Simulation(object):
    """
    Abstract simulator interface
    """

    @staticmethod
    def restoreFromFile(fileName):
        with open(fileName, 'rb') as f:
            return dill.load(f)

    __metaclass__ = ABCMeta  # declare as abstract class

    def __init__(self, name, solver, *args, **kwargs):
        self.name = name
        self.solver = solver

    @abstractmethod
    def getSetup(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def generateSimulationNodes(self, simulationNodes, read_nodes_from_file=False, parameters_file_name=None, parameters_setup_file_name=None):
        raise NotImplementedError("Should have implemented this")

    def prepareSolver(self):
        self.solver.prepare(self.parameters)

    @abstractmethod
    def prepareStatistic(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        """
        Call this function when you need to setup a parallel statistics
        """
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None, *args, **kwargs):
        raise NotImplementedError("Should have implemented this")

    def saveToFile(self, fileName):
        # save state file
        simFileName = fileName + '.sim'
        with open(simFileName, 'wb') as f:
            dill.dump(self, f)

    def saveParametersToFile(self, fileName):
        """
        This function saves parameters used to stimulate the model in a file
        """
        if hasattr(self, 'parameters') and self.parameters is not None:
            paramFileName = fileName + '.npy'
            np.save(paramFileName, self.parameters)
        else:
            print("No parameters to save")

    def get_simulation_parameters_shape(self):
        return self.parameters.shape
