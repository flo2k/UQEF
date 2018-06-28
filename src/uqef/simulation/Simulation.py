"""
Abstract base class for simulations like: Monte Carlo, or stochastic collocation

@author: Florian Kuenzner
"""

from abc import abstractmethod
from abc import ABCMeta

import dill


def restoreFromFile(statFileName):
    with open(statFileName, 'rb') as f:
        return dill.load(f)


class Simulation(object):
    """
    Abstract simulator interface
    """

    __metaclass__ = ABCMeta  # declare as abstract class
    
    def __init__(self, name, solver):
        self.name = name
        self.solver = solver

    @abstractmethod
    def getSetup(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def generateSimulationNodes(self, simulationNodes):
        raise NotImplementedError("Should have implemented this")
    
    def prepareSolver(self):
        self.solver.prepare(self.parameters)

    @abstractmethod
    def calculateStatistics(self, statistics, simulationNodes, original_runtime_estimator=None):
        raise NotImplementedError("Should have implemented this")

    def saveToFile(self, fileName):
        # save state file
        statFileName = fileName + '.sim'
        with open(statFileName, 'wb') as f:
            dill.dump(self, f)
