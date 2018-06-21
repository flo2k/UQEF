"""
Abstract base class for a model.

For each concrete model, a derivation of the Model class is done.

@author: Florian Kuenzner
"""

from abc import abstractmethod
from abc import ABCMeta


class Model(object):
    """
    Model is an abstract base class for concrete models.
    """

    __metaclass__ = ABCMeta  # declare as abstract class

    def __init__(self):
        pass

    @abstractmethod
    def prepare(self):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def assertParameter(self, parameter):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def normaliseParameter(self, parameters):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def run(self, i, parameter):
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def timesteps(self):
        raise NotImplementedError("Should have implemented this")
