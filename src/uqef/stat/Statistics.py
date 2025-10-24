"""
Abstract base class for statistics calculation

@author: Florian Kuenzner
"""

import os
import dill
import pickle
import numpy as np
import sys

dill.settings['recurse'] = True

class Statistics(object):
    """
    Abstract statistics interface
    """

    @staticmethod
    def restoreFromFile(fileName):
        with open(fileName, 'rb') as f:
            return dill.load(f)

    def __init__(self):
        self.timesteps = np.array([])
        pass

    def prepare(self, rawSamples, **kwargs):
        pass

    def set_nodes(self, simulationNodes):
        pass

    def prepareForMcStatistics(self, simulationNodes, numEvaluations, regression=False, order=None,
                               poly_normed=None, poly_rule=None, cross_truncation=1.0, regression_model_type=None, *args, **kwargs):
        pass

    def prepareForScStatistics(self, simulationNodes, order, poly_normed, poly_rule, regression=False, cross_truncation=1.0, regression_model_type=None, *args, **kwargs):
        pass

    def prepareForMcSaltelliStatistics(self, simulationNodes, numEvaluations, regression=False, order=None, 
                                        poly_normed=None, poly_rule=None, cross_truncation=1.0, *args, **kwargs):
        pass

    def prepareForEnsembleStatistics(self, simulationNodes, numEvaluations, *args, **kwargs):
        pass

    def calcStatisticsForMcParallel(self, chunksize=1, *args, **kwargs):
        pass

    def calcStatisticsForScParallel(self, chunksize=1, *args, **kwargs):
        pass

    def calcStatisticsForMcSaltelliParallel(self, chunksize=1, *args, **kwargs):
        pass

    def calcStatisticsForEnsembleParallel(self, chunksize=1, *args, **kwargs):
        pass

    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator, 
                            poly_normed=None, poly_rule=None, cross_truncation=1.0, regression_model=None,
                            *args, **kwargs):
        pass

    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator, 
                            poly_normed=None, poly_rule=None, cross_truncation=1.0, regression_model=None,
                            *args, **kwargs):
        pass

    def calcStatisticsForMcSaltelli(self, rawSamples, timesteps,
                                    simulationNodes, numEvaluations, order, regression, solverTimes,
                                    work_package_indexes, original_runtime_estimator, 
                                    poly_normed=None, poly_rule=None, cross_truncation=1.0,
                                    *args, **kwargs):
        pass

    def calcStatisticsForEnsemble(self, rawSamples=None, timesteps=None, simulationNodes=None, numEvaluations=None, solverTimes=None,
                                  work_package_indexes=None, original_runtime_estimator=None, *args, **kwargs):
        pass

    def printResults(self, timestep=-1, **kwargs):
        pass

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        pass

    def generateFileName(self,
                         fileName="", fileNameIdent="", directory="./",
                         fileNameIdentIsFullName=False):

        if not str(directory).endswith("/"):
            directory = str(directory) + "/"

        if fileName == "":
            fileName = os.path.splitext(sys.argv[0])[0]

        if fileNameIdentIsFullName:
            fileName = fileNameIdent
        else:
            fileName = directory + fileName
            if len(fileNameIdent) > 0:
                fileName = fileName + fileNameIdent

        return fileName

    def plotAnimation(self, timesteps, display=False,
                      fileName="", fileNameIdent="", directory="./",
                      fileNameIdentIsFullName=False, safe=True, **kwargs):
        pass

    def saveToFile(self,
                   fileName="", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):
        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)

        #save state file
        statFileName = fileName + '.stat'
        with open(statFileName, 'wb') as f:
            # dill.dump(self, f)
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    def saveAsNetCdf(self, timesteps,
                     fileName="", fileNameIdent="", directory="./",
                     fileNameIdentIsFullName=False):
        pass
