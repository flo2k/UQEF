"""
The TestModel is a very simple implementation of a Model that returns exactly the parameter value as the
VoI - Value of Interest. Mathematically: the identity!

@author: Florian Kuenzner
"""

from .Model import Model

import numpy as np
import time


class TestModel(Model):
    """
    A simple test model that sleeps the given parameter time!
    """

    def __init__(self):
        Model.__init__(self)

        self.t = range(0, 5)

    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i, parameter):
        print("{}: paramater: {}".format(i, parameter))

        value_of_interest = np.sum(parameter)
        #time.sleep(value_of_interest)

        return [value_of_interest]

    def timesteps(self):
        return self.t