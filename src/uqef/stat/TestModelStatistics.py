

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plotter
import chaospy as cp

import os
import sys
import pickle

from .Statistics import Statistics

class Samples(object):
    """
    Samples is a collection of the sampled results of a whole UQ simulation
    """

    def __init__(self, rawSamples, numTimeSteps):
        self.voi = []

        for sample in rawSamples:
            self.voi.append(sample)

        self.voi = np.array(self.voi)


class TestModelStatistics(Statistics):
    """
    TestModelStatistics calculates the statistics for the TestModel
    """

    def __init__(self):
        Statistics.__init__(self)

    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order,
                            regression,
                            solverTimes,
                            work_package_indexes, original_runtime_estimator,
                            *args, **kwargs):
        self.timesteps = timesteps
        self.numTimeSteps = len(self.timesteps)
        samples = Samples(rawSamples, self.numTimeSteps)

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.expansion.stieltjes(order, dist)
            self.qoi_gPCE = cp.fit_regression(P, nodes, samples.voi)
            self.calc_stats_for_gPCE(dist)
        else:
            ##extract the statistics
            # expectation value
            self.E_qoi = np.sum(samples.voi, 0) / numEvaluations

            # percentiles
            self.P10_qoi = np.percentile(samples.voi, 10, axis=0)
            self.P90_qoi = np.percentile(samples.voi, 90, axis=0)

            if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
                self.P10_qoi = self.P10_qoi[0]
                self.P90_qoi = self.P90_qoi[0]

            # variance
            self.Var_qoi = np.sum(samples.voi ** 2, 0) / numEvaluations - self.E_qoi ** 2

            # standard deviation
            self.StdDev_qoi = np.sqrt(self.Var_qoi)

    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, regression, poly_normed, poly_rule,
                            solverTimes, work_package_indexes, original_runtime_estimator,
                            *args, **kwargs):
        nodes = simulationNodes.distNodes
        weights = simulationNodes.weights
        dist = simulationNodes.joinedDists

        self.timesteps = timesteps
        self.numTimeSteps = len(self.timesteps)

        samples = Samples(rawSamples, self.numTimeSteps)

        P = cp.expansion.stieltjes(order, dist)

        if regression:
            self.qoi_gPCE = cp.fit_regression(P, nodes, samples.voi)
        else:
            self.qoi_gPCE = cp.fit_quadrature(P, nodes, weights, samples.voi)

        self.calc_stats_for_gPCE(dist)

    def calc_stats_for_gPCE(self, dist):
        ##extract the statistics
        # expectation value
        self.E_qoi = cp.E(self.qoi_gPCE, dist)

        # percentiles
        numPercSamples = 10 ** 5

        self.P10_qoi = cp.Perc(self.qoi_gPCE, 10, dist, numPercSamples)
        self.P90_qoi = cp.Perc(self.qoi_gPCE, 90, dist, numPercSamples)

        if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]

        # variance
        self.Var_qoi = cp.Var(self.qoi_gPCE, dist)

        # standard deviation
        self.StdDev_qoi = cp.Std(self.qoi_gPCE, dist)

        # sobol indices
        self.Sobol_m_qoi = cp.Sens_m(self.qoi_gPCE, dist)

        self.Sobol_m2_qoi = cp.Sens_m2(self.qoi_gPCE, dist)

        self.Sobol_t_qoi = cp.Sens_t(self.qoi_gPCE, dist)

    def calcStatisticsForMcSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator=None,
                            *args, **kwargs):
        # TODO: do some tests with separate implementation of Saltelli stats
        self.calcStatisticsForMc(rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator)

    def printResults(self, timestep=-1, printAllTimeSteps=False, **kwargs):
        resultTable = []
        resultTable.append(["qoi", self.E_qoi, self.StdDev_qoi, self.Var_qoi, self.P10_qoi, self.P90_qoi])

        string = tabulate(resultTable, headers=["QoI", "E", "StdDev", "Var", "P10_qoi", "P90_qoi"]) + "\n"
        string += "\n"

        return string

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)

        #####################################
        ### plot: mean + percentiles
        #####################################

        figure = plotter.figure(1, figsize=(13, 10))
        figure.canvas.manager.set_window_title('TestModel statistics')

        plotter.subplot(311)
        # plotter.title('mean')
        plotter.plot(self.timesteps, self.E_qoi, 'o', label='mean')
        plotter.fill_between(self.timesteps, self.P10_qoi, self.P90_qoi, facecolor='#5dcec6')
        plotter.plot(self.timesteps, self.P10_qoi, 'o', label='10th percentile')
        plotter.plot(self.timesteps, self.P90_qoi, 'o', label='90th percentile')
        plotter.xlabel('time (t) - seconds', fontsize=13)
        plotter.ylabel('testmodel value', fontsize=13)
        #plotter.xlim(0, 200)
        ymin, ymax = plotter.ylim()
        plotter.ylim(0, 20)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        plotter.subplot(312)
        # plotter.title('standard deviation')
        plotter.plot(self.timesteps, self.StdDev_qoi, 'o', label='std. dev.')
        plotter.xlabel('time (t) - seconds', fontsize=13)
        plotter.ylabel('standard deviation (pedestrians)', fontsize=13)
        #plotter.xlim(0, 200)
        plotter.ylim(0, 20)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        if hasattr(self, "Sobol_m_qoi"):
            plotter.subplot(313)
            sobol_labels = ["uncertain_param_1", "uncertain_param_2"]
            for i in range(np.size(self.Sobol_m_qoi, 0)):
                plotter.plot(self.timesteps, self.Sobol_m_qoi[i], 'o', label=sobol_labels[i])
            plotter.xlabel('time (t) - seconds', fontsize=13)
            plotter.ylabel('sobol indices', fontsize=13)
            #plotter.xlim(0, 200)
            plotter.ylim(-0.1, 1.1)
            plotter.legend()  # enable the legend
            plotter.grid(True)

        # save figure mean + variance
        pdfFileName = fileName + "_uq" + '.pdf'
        pngFileName = fileName + "_uq" + '.png'
        plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(pngFileName, format='png')

        if display:
            plotter.show()

        plotter.close()
