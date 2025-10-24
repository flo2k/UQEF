"""
Automatic evaluation of runtime statistics for any model

@author: Florian Kuenzner
"""

import numpy as np
import itertools
import more_itertools
from tabulate import tabulate
import matplotlib.pyplot as plotter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from .Statistics import Statistics

import os
import sys
import pickle
import csv

import chaospy as cp
from fileinput import filename


class Samples(object):
    """
    Samples is a collection of the sampled runtime results of a whole UQ simulation
    """

    def __init__(self, rawSamples, numTimeSteps):
        self.runtime = []

        for sample in rawSamples:
            self.runtime.append(sample)

        self.runtime = np.array(self.runtime)


class RuntimeStatistics(Statistics):
    """
    RuntimeStatistics calculates the runtime statistics for any model
    """
    
    def __init__(self):
        Statistics.__init__(self)

        self.runtime_estimator = None

    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator=None,
                            *args, **kwargs):
        self.timesteps = timesteps
        self.numTimeSteps = len(self.timesteps)
        self.solverTimes = solverTimes
        samples = Samples(self.solverTimes.T_i_S, self.numTimeSteps)

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.expansion.stieltjes(order, dist)
            self.GPCe_runtime = cp.fit_regression(P, nodes, samples.runtime)
            self.calc_stats_for_gPCE(dist)
        else:
            ##extract the statistics
            # expectation value
            self.E_runtime = np.sum(samples.runtime, 0) / numEvaluations

            # percentiles
            self.P5_runtime = np.percentile(samples.runtime, 5, axis=0)
            self.P95_runtime = np.percentile(samples.runtime, 90, axis=0)

    #        if len(self.P5_runtime) == 1:
    #            self.P5_runtime = self.P5_runtime[0]
    #            self.P95_runtime = self.P95_runtime[0]

            # variance
            self.Var_runtime = np.sum(samples.runtime ** 2, 0) / numEvaluations - self.E_runtime ** 2

            # standard deviation
            self.StdDev_runtime = np.sqrt(self.Var_runtime)

            #qoi dists
            self.qoi_dist_sampling = np.linspace(5, 45, 10**4, endpoint=True)

            qoi_dist_runtime = cp.GaussianKDE(samples.runtime)

            self.qoi_dist_runtime_pdf = np.zeros((len(self.qoi_dist_sampling)))
            self.qoi_dist_runtime_pdf = qoi_dist_runtime.pdf(self.qoi_dist_sampling)

        # real runtime
        self.real_runtime = samples.runtime.copy()

    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, regression, poly_normed, poly_rule,
                            solverTimes, work_package_indexes, original_runtime_estimator=None,
                            *args, **kwargs):
        nodes = simulationNodes.distNodes
        weights = simulationNodes.weights
        dist = simulationNodes.joinedDists
        self.solverTimes = solverTimes

        self.timesteps = timesteps
        self.numTimeSteps = len(self.timesteps)
        
        samples = Samples(self.solverTimes.T_i_S, self.numTimeSteps)
        
        P = cp.expansion.stieltjes(order, dist)

        if regression:
            self.GPCe_runtime = cp.fit_regression(P, nodes, samples.runtime)
        else:
            self.GPCe_runtime = cp.fit_quadrature(P, nodes, weights, samples.runtime)

        self.runtime_estimator = self.GPCe_runtime

        self.calc_stats_for_gPCE(dist)

        # qoi_dists
        delta = 2.0
        self.qoi_dist_sampling = np.linspace(samples.runtime.min() - delta, samples.runtime.max() + delta, 10**4, endpoint=True)
        print(self.qoi_dist_sampling)

        qoi_dist_runtime = cp.QoI_Dist(self.GPCe_runtime, dist, sample=5e4)
        self.qoi_dist_runtime_pdf = np.zeros((len(self.qoi_dist_sampling)))
        self.qoi_dist_runtime_pdf = qoi_dist_runtime.pdf(self.qoi_dist_sampling)
        #print "self.qoi_dist_runtime_pdf: " + str(self.qoi_dist_runtime_pdf)

        # real runtime
        self.real_runtime = samples.runtime.copy()

        self.T_i_SWP_worker = self.solverTimes.T_i_SWP_worker
        self.T_SWP_worker = self.solverTimes.T_SWP_worker

        self.T_Prop = self.solverTimes.T_Prop

        self.T_I = self.solverTimes.T_I
        self.T_C = self.solverTimes.T_C
        self.T_S_overhead = self.solverTimes.T_S_overhead

        # estimated runtime
        self.estimated_runtime = np.array([self.runtime_estimator(*node) for node in nodes.T])
        self.ET_i_S = self.estimated_runtime
        self.ET_i_S_errors = np.absolute(self.real_runtime - self.estimated_runtime)
        self.ET_i_S_errors_min = np.min(self.ET_i_S_errors)
        self.ET_i_S_errors_max = np.max(self.ET_i_S_errors)
        self.ET_i_S_errors_mean = np.mean(self.ET_i_S_errors)
        self.ET_i_S_errors_l2_norm = np.sqrt(np.sum(self.ET_i_S_errors**2)/len(self.ET_i_S_errors))
        self.ET_i_S_errors_rel = self.ET_i_S_errors / np.maximum(self.real_runtime, self.estimated_runtime)
        self.ET_i_S_errors_rel_min = np.min(self.ET_i_S_errors_rel)
        self.ET_i_S_errors_rel_max = np.max(self.ET_i_S_errors_rel)
        self.ET_i_S_errors_rel_mean = np.mean(self.ET_i_S_errors_rel)
        self.ET_i_S_errors_rel_l2_norm = np.sqrt(np.sum(self.ET_i_S_errors_rel ** 2) / len(self.ET_i_S_errors_rel))

        self.ET_i_SWP_i_worker = []
        for wp in work_package_indexes:
            self.ET_i_SWP_i_worker.append([self.estimated_runtime[wi] for wi in wp])

        self.ET_i_SWP_worker = []
        for i in range(0, len(self.ET_i_SWP_i_worker)):
            self.ET_i_SWP_worker.append(np.sum(self.ET_i_SWP_i_worker[i]) / self.solverTimes.parallel_solvers_per_work_package[i])
        self.ET_i_SWP_worker = np.asarray(self.ET_i_SWP_worker)

        self.ET_SWP_worker = self.ET_i_SWP_worker.max()

        self.ET_I = self.T_I
        self.ET_C = self.T_C
        self.ET_S_overhead = self.T_S_overhead

        self.ET_Prop = self.ET_SWP_worker + self.ET_C + self.ET_S_overhead

        # original estimated runtime
        if original_runtime_estimator is None:
            self.original_estimated_runtime = np.array([self.runtime_estimator(*node) for node in nodes.T])
        else:
            self.original_estimated_runtime = np.array([original_runtime_estimator(*node) for node in nodes.T])
        self.OET_i_S = self.original_estimated_runtime
        self.OET_i_S_errors = np.absolute(self.real_runtime - self.original_estimated_runtime)
        self.OET_i_S_errors_min = np.min(self.OET_i_S_errors)
        self.OET_i_S_errors_max = np.max(self.OET_i_S_errors)
        self.OET_i_S_errors_mean = np.mean(self.OET_i_S_errors)
        self.OET_i_S_errors_l2_norm = np.sqrt(np.sum(self.OET_i_S_errors**2))
        self.OET_i_S_errors_rel = self.OET_i_S_errors / np.maximum(self.real_runtime, self.original_estimated_runtime)
        self.OET_i_S_errors_rel_min = np.min(self.OET_i_S_errors_rel)
        self.OET_i_S_errors_rel_max = np.max(self.OET_i_S_errors_rel)
        self.OET_i_S_errors_rel_mean = np.mean(self.OET_i_S_errors_rel)
        self.OET_i_S_errors_rel_l2_norm = np.sqrt(np.sum(self.OET_i_S_errors_rel ** 2) / len(self.OET_i_S_errors_rel))

        self.OET_i_SWP_i_worker = []
        for wp in work_package_indexes:
            self.OET_i_SWP_i_worker.append([self.original_estimated_runtime[wi] for wi in wp])

        self.OET_i_SWP_worker = []
        for i in range(0, len(self.OET_i_SWP_i_worker)):
            self.OET_i_SWP_worker.append(np.sum(self.OET_i_SWP_i_worker[i]) / self.solverTimes.parallel_solvers_per_work_package[i])
        self.OET_i_SWP_worker = np.asarray(self.OET_i_SWP_worker)

        self.OET_SWP_worker = self.OET_i_SWP_worker.max()

        self.OET_I = self.T_I
        self.OET_C = self.T_C
        self.OET_S_overhead = self.T_S_overhead

        self.OET_Prop = self.OET_SWP_worker + self.OET_C + self.OET_S_overhead

    def calc_stats_for_gPCE(self, dist):
        ## extract the statistics
        # expectation value
        self.E_runtime = cp.E(self.GPCe_runtime, dist)

        # percentiles
        numPercSamples = 10 ** 5

        self.P5_runtime = cp.Perc(self.GPCe_runtime, 5, dist, numPercSamples)
        self.P95_runtime = cp.Perc(self.GPCe_runtime, 95, dist, numPercSamples)

        #        if len(self.P5_runtime) == 1:
        #            self.P5_runtime = self.P5_runtime[0]
        #            self.P95_runtime = self.P95_runtime[0]

        # variance
        self.Var_runtime = cp.Var(self.GPCe_runtime, dist)

        # standard deviation
        self.StdDev_runtime = cp.Std(self.GPCe_runtime, dist)

        # sobol indices
        self.Sobol_m_runtime = cp.Sens_m(self.GPCe_runtime, dist)
        self.Sobol_m_runtime_int = 1 - np.sum(self.Sobol_m_runtime)

        self.Sobol_m2_runtime = cp.Sens_m2(self.GPCe_runtime, dist)

        self.Sobol_t_runtime = cp.Sens_t(self.GPCe_runtime, dist)

    def calcStatisticsForMcSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator=None,
                            *args, **kwargs):
        # TODO: do some tests with separate implementation of Saltelli stats
        self.calcStatisticsForMc(rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator)

    def printResults(self, timestep=-1, printAllTimeSteps=False, **kwargs):
        # print results
        resultTable = []
        resultTable.append(["E_runtime", self.E_runtime, self.StdDev_runtime, self.Var_runtime, self.P5_runtime, self.P95_runtime])

        string = tabulate(resultTable, headers=["QoI", "E", "StdDev", "Var", "P5", "P95"], floatfmt="f") + "\n"

        string += "\n"

        if hasattr(self, "Sobol_m_runtime"):
            resultTable = []
            resultTable.append(["First order", str(self.Sobol_m_runtime)])
            resultTable.append(["First order_Interaction", self.Sobol_m_runtime_int])
            resultTable.append(["Second order", str(self.Sobol_m2_runtime)])
            resultTable.append(["Total order", str(self.Sobol_t_runtime)])

            string += "Sensitivity Indices:" + "\n"
            string += tabulate(resultTable,
                               headers=["Index", "value"],
                               floatfmt="f") + "\n"
            string += "\n"

        if hasattr(self, "estimated_runtime") and hasattr(self, "estimated_runtime") and hasattr(self, "original_estimated_runtime"):
            resultTable = []
            #resultTable.append(["sum(T_i_S)",     self.real_runtime.sum(), self.estimated_runtime.sum(), self.original_estimated_runtime.sum(), (self.real_runtime.sum() - self.estimated_runtime.sum()), (self.real_runtime.sum() - self.original_estimated_runtime.sum())])

            resultTable.append(["T_i_SWP_worker", self.T_i_SWP_worker.tolist(), self.ET_i_SWP_worker.tolist(), self.OET_i_SWP_worker.tolist(), (self.T_i_SWP_worker - self.ET_i_SWP_worker).tolist(), (self.T_i_SWP_worker - self.OET_i_SWP_worker).tolist()])
            resultTable.append(["T_SWP_worker"  , self.T_SWP_worker.tolist()  , self.ET_SWP_worker.tolist()  , self.OET_SWP_worker.tolist()  , (self.T_SWP_worker - self.ET_SWP_worker).tolist()    , (self.T_SWP_worker - self.OET_SWP_worker).tolist()    ])
            resultTable.append(["T_Prop"        , self.T_Prop.tolist()        , self.ET_Prop.tolist()        , self.OET_Prop.tolist()        , (self.T_Prop - self.ET_Prop).tolist()                , (self.T_Prop - self.OET_Prop).tolist()                ])
            resultTable.append(["T_I"           , self.T_I.tolist()           , self.ET_I.tolist()           , self.OET_I.tolist()           , (self.T_I - self.ET_I).tolist()                      , (self.T_I - self.OET_I).tolist()                      ])
            resultTable.append(["T_C"           , self.T_C                    , self.ET_C                    , self.OET_C                    , (self.T_C - self.ET_C)                               , (self.T_C - self.OET_C)                               ])
            resultTable.append(["T_S_overhead"  , self.T_S_overhead           , self.ET_S_overhead           , self.OET_S_overhead           , (self.T_S_overhead - self.ET_S_overhead)             , (self.T_S_overhead - self.OET_S_overhead)             ])

            string += tabulate(resultTable, headers=["type", "runtime", "estimated", "org_estimated", "diff: T-E", "diff: T-OE"], floatfmt="f") + "\n"

            string += "\n"

        if hasattr(self, "estimated_runtime"):
            resultTable = []
            resultTable.append(["absolute error min"    , self.ET_i_S_errors_min    , self.OET_i_S_errors_min])
            resultTable.append(["absolute error max"    , self.ET_i_S_errors_max    , self.OET_i_S_errors_max])
            resultTable.append(["absolute error mean"   , self.ET_i_S_errors_mean   , self.OET_i_S_errors_mean])
            resultTable.append(["absolute error l2 norm", self.ET_i_S_errors_l2_norm, self.OET_i_S_errors_l2_norm])
            resultTable.append(["relative error min",     self.ET_i_S_errors_rel_min,     self.OET_i_S_errors_rel_min])
            resultTable.append(["relative error max",     self.ET_i_S_errors_rel_max,     self.OET_i_S_errors_rel_max])
            resultTable.append(["relative error mean",    self.ET_i_S_errors_rel_mean,    self.OET_i_S_errors_rel_mean])
            resultTable.append(["relative error l2 norm", self.ET_i_S_errors_rel_l2_norm, self.OET_i_S_errors_rel_l2_norm])

            string += "Errors:" + "\n"
            string += tabulate(resultTable,
                               headers=["Error", "ET", "OET"],
                               floatfmt="f") + "\n"
            string += "\n"

            if hasattr(self.solverTimes, "t_estimate_runtime"):
                resultTable = []
                resultTable.append(["t_estimate_runtime", self.solverTimes.t_estimate_runtime])
                resultTable.append(["t_wp_creation", self.solverTimes.t_wp_creation])
                resultTable.append(["t_estimate_restore_order", self.solverTimes.t_estimate_restore_order])

            string += tabulate(resultTable, headers=["type", "runtime"], floatfmt="f") + "\n"
            string += "\n"

        return string

    def printCsv(self,
                 fileName="", fileNameIdent="", directory="./", 
                 fileNameIdentIsFullName=False):
        
        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)
        headerRow = ["no", "nw", "so", "sw", "n", "s"]
        
        return
        
    def plotResults(self, timestep=-1, display=False, 
                    fileName="", fileNameIdent="", directory="./", 
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)

        # #####################################
        # ### plot: mean + percentiles
        # #####################################
        #
        # figure = plotter.figure(1, figsize=(13,5))
        # figure.canvas.manager.set_window_title('UQ: Train Station')
        #
        # figure = plotter.figure(1, figsize=(18, 5))
        # figure.canvas.manager.set_window_title('UQ: Pedestrian Evacuation (Vadere)')
        #
        # plotter.subplot(131)
        # # plotter.title('mean')
        # plotter.plot(self.timesteps, self.E_ev1, label='mean')
        # # plotter.plot(time_points, self.E_ev2, label='E_ev2')
        # plotter.fill_between(self.timesteps, self.P10_ev1, self.P90_ev1, facecolor='#5dcec6')
        # # plotter.fill_between(time_points, self.P10_ev2, self.P90_ev2, facecolor='#5dcec6')
        # plotter.plot(self.timesteps, self.P10_ev1, label='10th percentile')
        # plotter.plot(self.timesteps, self.P90_ev1, label='90th percentile')
        # # plotter.plot(time_points, self.P10_ev2, label='P10_ev2')
        # # plotter.plot(time_points, self.P90_ev2, label='P90_ev2')
        # plotter.xlabel('time (t) - seconds', fontsize=13)
        # plotter.ylabel('pedestrians', fontsize=13)
        # plotter.xlim(0, 200)
        # ymin, ymax = plotter.ylim()
        # plotter.ylim(0, 102)
        # plotter.legend()  # enable the legend
        # plotter.grid(True)
        #
        # plotter.subplot(132)
        # # plotter.title('standard deviation')
        # plotter.plot(self.timesteps, self.StdDev_ev1, label='std. dev.')
        # # plotter.plot(time_points, self.Var_ev1, label='Var_ev1')
        # # plotter.plot(time_points, self.Var_ev2, label='Var_ev2')
        # # plotter.plot(time_points, E_pX.T[0], label='p')
        # # plotter.fill_between(time_points, P10_pX.T[0], P90_pX.T[0], facecolor='#5dcec6')
        # # plotter.plot(time_points, P10_pX.T[0], label='P10')
        # # plotter.plot(time_points, P90_pX.T[0], label='P90')
        # plotter.xlabel('time (t) - seconds', fontsize=13)
        # plotter.ylabel('standard deviation (pedestrians)', fontsize=13)
        # plotter.xlim(0, 200)
        # plotter.ylim(0, 102)
        # plotter.legend()  # enable the legend
        # plotter.grid(True)
        #
        # if hasattr(self, "Sobol_m_ev1"):
        #     plotter.subplot(133)
        #     sobol_labels = ["familyPercentage", "childrenSpeed", "parentSpeed"]
        #     for i in range(np.size(self.Sobol_m_ev1, 0)):
        #         plotter.plot(self.timesteps, self.Sobol_m_ev1[i], label=sobol_labels[i])
        #     plotter.xlabel('time (t) - seconds', fontsize=13)
        #     plotter.ylabel('sobol indices', fontsize=13)
        #     plotter.xlim(0, 200)
        #     plotter.ylim(-0.1, 1.1)
        #     plotter.legend()  # enable the legend
        #     plotter.grid(True)
        #
        # #save figure mean + variance
        # pdfFileName = fileName + "_uq" + '.pdf'
        # pngFileName = fileName + "_uq" + '.png'
        # plotter.savefig(pdfFileName, format='pdf')
        # plotter.savefig(pngFileName, format='png')
        #
        # if display:
        #     plotter.show()
        #
        # plotter.close()

        #####################################
        ### plot: qoi_dist
        #####################################

        if hasattr(self, "qoi_dist_runtime_pdf"):
            figure = plotter.figure(1, figsize=(12, 4))
            figure.canvas.manager.set_window_title('UQ: Pedestrian Evacuation (Vadere)')

            # qoi_dist
            plotter.subplot(111)

            # time_points_for_qoi_dist = self.timesteps[np.where(self.timesteps <= 200)]
            plotter.plot(self.qoi_dist_sampling, self.qoi_dist_runtime_pdf, label="runtime pdf")
            # plotter.plot(self.qoi_dist_sampling, self.qoi_dist_waitingareaS_pdf.T[8], label="T=20, S")
            plotter.xlabel('runtime (s)', fontsize=13)
            plotter.ylabel('probability', fontsize=13)
            # xlim = plotter.xlim(0, 130)
            #plotter.xlim(0, 0.5)
            # plotter.ylim(0, 1)
            plotter.legend()  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.13, top=0.95, left=0.1, right=0.99)

            # save figure qoi_dist
            pdfFileName = fileName + "_qoi_dist" + '.pdf'
            pngFileName = fileName + "_qoi_dist" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: real_runtime
        #####################################

        figure = plotter.figure(1, figsize=(6, 4))
        figure.canvas.manager.set_window_title('UQ: runtime')

        fontsize = 15
        plotter.rc('font', family='serif', size=fontsize)

        # runtime as it is
        ax = plotter.subplot(111)
        ax.set_title("(a) Real runtime $T^i_S$")
        ax.plot(np.arange(1, len(self.real_runtime) + 1), self.real_runtime, 'o', markerfacecolor="w", label="$T^i_S$ (real)")
        # plotter.plot(self.original_estimated_runtime, 'o', label="org predicted runtime")
        plotter.xlabel('collocation points ($z_i$)')
        plotter.ylabel('runtime (s)')
        plotter.legend(fontsize=fontsize-2)  # enable the legend
        plotter.grid(True)

        # Plotter settings
        plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

        # save figure qoi_dist
        pdfFileName = fileName + "_real_runtime" + '.pdf'
        pngFileName = fileName + "_real_runtime" + '.png'
        plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(pngFileName, format='png')

        if display:
            plotter.show()

        plotter.close()

        #####################################
        ### plot: real_runtime org
        #####################################
        if hasattr(self, "original_estimated_runtime"):
            figure = plotter.figure(1, figsize=(6, 4))
            figure.canvas.manager.set_window_title('UQ: runtime')

            fontsize = 15
            plotter.rc('font', family='serif', size=fontsize)

            # runtime as it is
            ax = plotter.subplot(111)

            ax.set_title(r"(b) Predicted runtime $\mathbb{T}^i_S$")
            ax.plot(np.arange(1, len(self.real_runtime) + 1), self.real_runtime, 'o', markerfacecolor="w", label="$T^i_S$ (real)")
            ax.plot(np.arange(1, len(self.original_estimated_runtime) + 1), self.original_estimated_runtime, 'go', label=r"$\mathbb{T}^i_S$ (predicted)")
            # plotter.plot(np.sort(self.real_runtime), 'o', label="real runtime")
            # plotter.plot(np.sort(self.estimated_runtime), 'go', label="predicted runtime")
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel('runtime (s)')
            plotter.legend(fontsize=fontsize - 2)  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_org" + '.pdf'
            pngFileName = fileName + "_real_runtime_org" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: real_runtime org comparison
        #####################################
        if hasattr(self, "original_estimated_runtime"):
            figure = plotter.figure(1, figsize=(12, 4))
            figure.canvas.manager.set_window_title('UQ: runtime')

            fontsize = 15
            plotter.rcParams.update({'font.size': fontsize})

            # runtime as it is
            ax = figure.add_subplot(121)

            plotter.title("(a) Real runtime $T^i_S$")
            plotter.plot(np.arange(1, len(self.real_runtime)+1), self.real_runtime, 'o', markerfacecolor="w", label="$T^i_S$ (real)")
            # plotter.plot(self.original_estimated_runtime, 'o', label="org predicted runtime")
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel('runtime (s)')
            plotter.legend(fontsize=fontsize-2)  # enable the legend
            plotter.grid(True)

            # sorted runtime
            ax = figure.add_subplot(122)

            plotter.title(r"(b) Predicted runtime $\mathbb{T}^i_S$")
            plotter.plot(np.arange(1, len(self.real_runtime)+1), self.real_runtime, 'o', markerfacecolor="w", label="$T^i_S$ (real)")
            plotter.plot(np.arange(1, len(self.original_estimated_runtime)+1), self.original_estimated_runtime, 'go', label=r"$\mathbb{T}^i_S$ (predicted)")
            # plotter.plot(np.sort(self.real_runtime), 'o', label="real runtime")
            # plotter.plot(np.sort(self.estimated_runtime), 'go', label="predicted runtime")
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel('runtime (s)')
            plotter.legend(fontsize=fontsize-2)  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.25, hspace=0.2, bottom=0.15, top=0.92, left=0.1, right=0.99)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_org_comparison" + '.pdf'
            pngFileName = fileName + "_real_runtime_org_comparison" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime_worker
        #####################################
        if hasattr(self, "T_i_SWP_worker"):
            figure = plotter.figure(1, figsize=(6, 4))
            figure.canvas.manager.set_window_title('UQ: runtime worker')

            fontsize = 15
            plotter.rc('font', family='serif', size=fontsize)

            # runtime as it is
            ax = plotter.subplot(111)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plotter.plot(range(1, self.T_i_SWP_worker.size+1),   self.T_i_SWP_worker,   'o', label="$T_{WP_j}$")
            #plotter.plot(range(1, self.ET_i_SWP_worker.size+1),  self.ET_i_SWP_worker,  'o', label="$\mathbb{T}_{WP_j}$ (ET_i_SWP_worker)")
            #plotter.plot(range(1, self.OET_i_SWP_worker.size+1), self.OET_i_SWP_worker, 'o', label="$\mathbb{T}_{WP_j}$ (OET_i_SWP_worker)")
            plotter.yscale('linear')
            plotter.ylim(0, max(self.T_SWP_worker.max(), self.ET_SWP_worker.max(), self.OET_SWP_worker.max()) + 10)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('worker ($WP_j$)')
            plotter.ylabel('runtime (s)')
            plotter.legend(loc="lower right", fontsize=fontsize-2)  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_worker" + '.pdf'
            pngFileName = fileName + "_real_runtime_worker" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime_worker org
        #####################################
        if hasattr(self, "T_i_SWP_worker"):
            figure = plotter.figure(1, figsize=(6, 4))
            figure.canvas.manager.set_window_title('UQ: runtime worker')

            # runtime as it is
            ax = plotter.subplot(111)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plotter.plot(range(1, self.T_i_SWP_worker.size+1),   self.T_i_SWP_worker,   'o', label="$T_{WP_j}$")
            plotter.plot(range(1, self.OET_i_SWP_worker.size+1), self.OET_i_SWP_worker, 'o', label=r"$\mathbb{T}_{WP_j}$")
            plotter.yscale('linear')
            plotter.ylim(0, max(self.T_SWP_worker.max(), self.ET_SWP_worker.max(), self.OET_SWP_worker.max()) + 10)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('worker ($WP_j$)')
            plotter.ylabel('runtime (s)')
            plotter.legend(loc="lower right", fontsize=fontsize-2)  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_worker_org" + '.pdf'
            pngFileName = fileName + "_real_runtime_worker_org" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime estimation errors org prediction
        #####################################
        if hasattr(self, "OET_i_S_errors"):
            figure = plotter.figure(1, figsize=(12, 4))
            figure.canvas.manager.set_window_title('UQ: runtime estimation error')

            fontsize = 15
            plotter.rcParams.update({'font.size': fontsize})

            # runtime as it is
            ax = figure.add_subplot(121)
            plotter.title("(a) Absolute error $er_i$")
            plotter.plot(range(1, self.OET_i_S_errors.size+1), self.OET_i_S_errors, 'o', label="$er_i$ (absolute error)")
            plotter.yscale('linear')
            plotter.ylim(0, self.OET_i_S_errors.max())
            # ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel('absolute error $er_i$ (s)')
            #plotter.legend(loc='upper left')  # enable the legend
            plotter.grid(True)

            ax = figure.add_subplot(122)
            plotter.title("(b) Relative error $er_{i,rel}$")
            plotter.plot(range(1, self.OET_i_S_errors_rel.size+1), self.OET_i_S_errors_rel, 'o', label="$er_{i,rel}$ (relative error)")
            #plotter.yscale('log')
            #plotter.ylim(0, 1)
            print(self.OET_i_S_errors_rel)
            # ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel('relative error $er_{i,rel}$')
            #plotter.legend(loc='upper left')  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.25, hspace=0.2, bottom=0.15, top=0.92, left=0.1, right=0.99)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_estimation_error_org_prediciton" + '.pdf'
            pngFileName = fileName + "_real_runtime_estimation_error_org_prediciton" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime estimation absolute errors org prediction
        #####################################
        if hasattr(self, "OET_i_S_errors"):
            figure = plotter.figure(1, figsize=(6, 4))
            figure.canvas.manager.set_window_title('UQ: runtime estimation error')

            fontsize = 15
            plotter.rc('font', family='serif', size=fontsize)

            # runtime as it is
            ax = figure.add_subplot(111)
            ax.set_title(r"(a) Absolute error $\epsilon r_i$")
            ax.plot(range(1, self.OET_i_S_errors.size + 1), self.OET_i_S_errors, 'o', label="$er_i$ (absolute error)")
            plotter.yscale('linear')
            plotter.ylim(0, self.OET_i_S_errors.max())
            # ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel(r'absolute error $\epsilon r_i$ (s)')
            # plotter.legend(loc='upper left')  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_estimation_absolute_error_org_prediciton" + '.pdf'
            pngFileName = fileName + "_real_runtime_estimation_absolute_error_org_prediciton" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime estimation relative errors org prediction
        #####################################
        if hasattr(self, "OET_i_S_errors_rel"):
            figure = plotter.figure(1, figsize=(6, 4))
            figure.canvas.manager.set_window_title('UQ: runtime estimation error')

            fontsize = 15
            plotter.rc('font', family='serif', size=fontsize)

            ax = figure.add_subplot(111)
            ax.set_title(r"(b) Relative error $\epsilon r_{i,rel}$")
            ax.plot(range(1, self.OET_i_S_errors_rel.size + 1), self.OET_i_S_errors_rel, 'o', label="$er_{i,rel}$ (relative error)")
            # plotter.yscale('log')
            # plotter.ylim(0, 1)
            print(self.OET_i_S_errors_rel)
            # ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plotter.xlabel('collocation points ($z_i$)')
            plotter.ylabel(r'relative error $\epsilon r_{i,rel}$')
            # plotter.legend(loc='upper left')  # enable the legend
            plotter.grid(True)

            # Plotter settings
            plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.15, top=0.92, left=0.18, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_real_runtime_estimation_relative_error_org_prediciton" + '.pdf'
            pngFileName = fileName + "_real_runtime_estimation_relative_error_org_prediciton" + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

        #####################################
        ### plot: runtime histogram
        #####################################

        figure = plotter.figure(1, figsize=(12, 4))
        figure.canvas.manager.set_window_title('UQ: runtime historgram')

        # runtime as it is
        plotter.subplot(111)

        plotter.hist(self.real_runtime, histtype='bar', rwidth=0.96)
        plotter.xlabel('time (seconds)', fontsize=13)
        #plotter.ylabel('runtime (s)', fontsize=13)
        #plotter.legend()  # enable the legend
        #plotter.grid(True)

        # Plotter settings
        plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.13, top=0.95, left=0.1, right=0.99)

        # save figure qoi_dist
        pdfFileName = fileName + "_runtime_histogram" + '.pdf'
        pngFileName = fileName + "_runtime_histogram" + '.png'
        plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(pngFileName, format='png')

        if display:
            plotter.show()

        plotter.close()

    def saveRuntimeData(self,
                        fileName="", fileNameIdent="", directory="./",
                        fileNameIdentIsFullName=False,
                        **kwargs
                        ):
        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)

        # save runtime file
        runtimeFileName = fileName + '_runtimes_' + '.csv'
        with open(runtimeFileName, 'w') as f:
            writer = csv.writer(f)
            for r in self.real_runtime:
                writer.writerow([r])