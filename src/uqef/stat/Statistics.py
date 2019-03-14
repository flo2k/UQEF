"""
Abstract base class for statistics calculation

@author: Florian Kuenzner
"""

import os
import dill
import numpy as np


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
        
    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, solverTimes,
                            work_package_indexes, original_runtime_estimator):
        pass

    def calcStatisticsForSc(self, rawSamples, timestamps,
                            simulationNodes, order, solverTimes,
                            work_package_indexes, original_runtime_estimator=None):
        pass

    def printResults(self, timestep=-1):
        pass

    def plotResults(self, timestep=-1, display=False, 
                    fileName="", fileNameIdent="", directory="./", 
                    fileNameIdentIsFullName=False, safe=True):
        pass

    def generateFileName(self, 
                         fileName="", fileNameIdent="", directory="./",
                         fileNameIdentIsFullName=False):
        if not directory.endswith("/"):
            directory = directory + "/"
        
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
                      fileNameIdentIsFullName=False, safe=True):
        pass

    def saveToFile(self,
                   fileName="", fileNameIdent="", directory="./", 
                   fileNameIdentIsFullName=False):
        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)
        
        #save state file
        statFileName = fileName + '.stat'
        with open(statFileName, 'wb') as f:
            dill.dump(self, f)

    def saveAsNetCdf(self, timesteps,
                     fileName="", fileNameIdent="", directory="./", 
                     fileNameIdentIsFullName=False):
        pass
