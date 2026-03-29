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

        self.t = range(1, 2)

    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters):
        results = []
        for ip in range(0, len(i_s)):
            i = i_s[ip]
            parameter = parameters[ip]
            #print("{}: paramater: {}".format(i, parameter))

            start = time.time()

            # calc value of interest
            value_of_interest = np.sum(parameter)
            #time.sleep(value_of_interest)

            # measure runtime
            end = time.time()
            runtime = end - start

            # collect the output
            results.append((value_of_interest, runtime))

        return results

    def timesteps(self):
        return self.t