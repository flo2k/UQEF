'''
Created on 11.12.2015

@author: flo
'''

import numpy as np
import time
import math

import uqef

class RuntimeSimModel(uqef.model.Model):
    """
    RuntimeSimModel is a test model for simulate the runtime under uncertainty
    """

    def __init__(self, modelMasterPath, modelPath, masterScenarioFile, simulate_wait, variant=1, time_mul_factor=1.0, runtimesim_intrinsic_error=False):
        uqef.model.Model.__init__(self)
        
        # path
        self.modelMasterPath = modelMasterPath
        self.modelPath = modelPath
        self.scenariosPath = modelPath + "/" + "scenarios"
        self.masterConfigFile = modelMasterPath + "/scenarios_master/" + masterScenarioFile
        self.outputPath = modelPath + "/" + "output"

        self.variant = variant

        # config
        self.maxSimTime = 1
        self.dt =1
        self.numTimeSteps = 1
        
        # time_points
        self.time_points = np.zeros(1)

        self.simulate_wait = simulate_wait

        # time stretch
        self.time_mul_factor = time_mul_factor

        #intrinsic error
        self.runtimesim_intrinsic_error = runtimesim_intrinsic_error

    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter
 
    def run(self, i_s, parameters):
        results = []
        for ip in range(0, len(i_s)):
            i = i_s[ip]

            # parameters
            parameter = parameters[ip]
            familyPercentage = parameter[0]
            childrenSpeed = parameter[1]
            parentSpeed = parameter[2]
            seed = parameter[3]
            simTimeStepLength = parameter[4]

            start_time = time.time()

            #time.sleep(0.1)
            #time.sleep(0.1*math.log10(i+1))
            #time.sleep(random.random())

            # factor = 2.0
            # time.sleep(abs(familyPercentage) * 0.3 * factor)
            # time.sleep(abs(childrenSpeed) * 2.0 * factor)
            # time.sleep(abs(parentSpeed) * 0.3 * factor)
            #
            # end_time = time.time()
            # runtime = end_time - start_time
            # print "%d: takes %f secs" % (i, runtime)

            if self.variant == 1:
                # 1. Simple model to aim VADERE runtime behavior
                runtime = 0.0
                factor = 2.0
                ##runtime += abs(familyPercentage) * 10.0 * factor
                runtime += np.exp(abs(familyPercentage) * 5.0) * factor
                runtime += max(childrenSpeed, 0) * 1.0 * factor
                runtime += abs(parentSpeed) * 0.2 * factor
                ##runtime += np.random.rand(1)[0] # add some random factor
                ##runtime = runtime * 10.0
                runtime = runtime * self.time_mul_factor
            elif self.variant == 2:
                # 2. Model with discontinuity
                # Nicholas Zabarras Paper: „Sparse grid collocation schemes for stochastic natural convection problems“
                # e^(-x^2 + 2*sign(y))
                runtime = math.exp(-familyPercentage ** 2 + 2 * np.sign(childrenSpeed)) + parentSpeed
                #runtime = math.exp(-parentSpeed ** 2 + 2 * np.sign(familyPercentage)) + childrenSpeed
                #runtime = math.exp(-childrenSpeed ** 2 + 2 * np.sign(parentSpeed)) + familyPercentage

            if self.runtimesim_intrinsic_error:
                runtime += np.random.normal(0, 1)

            if not self.simulate_wait:
                start_time = time.time()
                time.sleep(runtime)

                end_time = time.time()
                runtime = end_time - start_time

            print("{:d}: takes {:f} secs".format(i, runtime))

            # collect the output
            results.append((runtime, runtime))

        return results

    def timesteps(self):
        return self.time_points
