"""
Created on 10.05.2015

@author: flo
"""

import chaospy as cp
import itertools
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plotter
import seaborn as sns
import json
import dill
import pickle
import psutil
import os
import sys
import pandas as pd

# this lines are added so that the line [dill.dump(self.parameters, f)] will work
import copyreg
import zipimport
copyreg.pickle(zipimport.zipimporter, lambda x: (x.__class__, (x.archive, )))

class Nodes(object):
    """
    Nodes represents the nodes and parameters for a UQ simulation
    """

    @staticmethod
    def restoreFromFile(fileName):
        with open(fileName, 'rb') as f:
            return dill.load(f)

    def __init__(self, nodeNames):
        """

        :param nodeNames: list of nodes names
        Important note:
        * nodes is a list of samples from standard distributions, used for e.g., generating polynomials
        * parameters on the other hand list of samples from a user specified distributions,
                     used for stimulating/forcing the model

        """
        self.nodeNames = nodeNames
        self.values = {}
        self.dists = {}
        self.joinedDists = []
        self.distNodes = []
        self.weights = []
        self.nodes = []
        self.numSamplesOrScDim = None

        self.standardDists = {}
        self.joinedStandardDists = []
        self._performTransformation = False
        self.parameters = None

        self.transformationParameters = {}
        self.transformationFunctions = {}

    def setValue(self, nodeName, value):
        self.assertNodeName(nodeName)

        self.values[nodeName] = value

    def setDist(self, nodeName, dist):
        self.assertNodeName(nodeName)

        self.dists[nodeName] = dist

    def setStandardDist(self, nodeName, dist):
        self.assertNodeName(nodeName)

        self.standardDists[nodeName] = dist

    def setTransformation(self):
        self._performTransformation = True

    def setTransformationParameters(self, nodeName, parametersTuple, transformationFunc):
        self.assertNodeName(nodeName)

        self._performTransformation = True
        self.transformationParameters[nodeName] = parametersTuple
        self.transformationFunctions[nodeName] = transformationFunc

    def assertNodeName(self, nodeName):
        assert nodeName in self.nodeNames, "name of node " + nodeName + " not registered."

    def assertConfiguration(self):
        numRegisteredNodes = len(self.nodeNames)
        numValues = len(self.values)
        numDists = len(self.dists)

        assert (numValues + numDists) == numRegisteredNodes, "not enough values registered"

    def getDistNodeNames(self):
        distNodeNames = [nodeName for nodeName in self.nodeNames if nodeName in self.dists]
        return distNodeNames

    def generateNodesForMC(self, numSamples, rule="R"):
        if self.numSamplesOrScDim == numSamples:
            return self.nodes

        self.assertConfiguration()
        self.numSamplesOrScDim = numSamples

        #order the distributes to get a defined order
        orderdDists = []
        orderdDistsNames = []
        orderdStandardDistsNames = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
                if self._performTransformation:
                    orderdStandardDistsNames.append(self.standardDists[nameOfNode])

        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            if self._performTransformation:
                self.joinedStandardDists = cp.J(*orderdStandardDistsNames)
                distNodes = self.joinedStandardDists.sample(size=numSamples, rule=rule).round(4)
                self.distNodes = distNodes
            else:
                #distNodes = self.joinedDists.sample(numSamples, rule=rule).round(4)
                #distNodes = cp.generate_samples(order=numSamples, domain=self.joinedDists, rule=rule).round(4)
                distNodes = self.joinedDists.sample(size=numSamples, rule=rule).round(4)
                self.distNodes = distNodes

        nodes = []

        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]

            if nameOfNode in self.values:
                nodes.append([self.values[nameOfNode]]*numSamples)

            if nameOfNode in self.dists:
                if len(self.dists) == 1:
                    nodes.append(distNodes)
                else:
                    nodes.append(distNodes[orderdDistsNames.index(nameOfNode)])

        self.nodes = np.array(nodes)
        self.weights = np.array(self.weights)  # MC has no weights, but after generation, we want a array

        if self._performTransformation:
            # self.parameters = self.transformParameters(orderdDistsNames, self.nodes)
            self.parameters = Nodes.transformSamples(self.nodes, self.joinedStandardDists, self.joinedDists)
        else:
            self.parameters = self.nodes

        return self.nodes, self.parameters

    def generateNodesForSC(self, numCollocationPointsPerDim, rule="G", sparse=False):

        if self.numSamplesOrScDim == numCollocationPointsPerDim:
            return self.nodes, self.weights

        self.numSamplesOrScDim = numCollocationPointsPerDim

        orderdDists = []
        orderdDistsNames = []
        orderdStandardDistsNames = []
        # self.joinedDists = []
        # self.distNodes = []
        # self.weights = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
                if self._performTransformation:
                    orderdStandardDistsNames.append(self.standardDists[nameOfNode])

        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            self.__save__cpu_affinity()
            growth = True if (rule == "c" and not sparse) else False  # according to: https://github.com/jonathf/chaospy/issues/139


            if self._performTransformation:
                self.joinedStandardDists = cp.J(*orderdStandardDistsNames)
                self.distNodes, self.weights = cp.generate_quadrature(numCollocationPointsPerDim,
                                                                      self.joinedStandardDists,
                                                                      rule=rule,
                                                                      growth=growth,
                                                                      sparse=sparse)
            else:
                self.distNodes, self.weights = cp.generate_quadrature(numCollocationPointsPerDim,
                                                                      self.joinedDists,
                                                                      rule=rule,
                                                                      growth=growth,
                                                                      sparse=sparse)
            self.__restore__cpu_affinity()

        nodes = []
        if len(self.distNodes) == 0:
            numNodes = numCollocationPointsPerDim
        else:
            numNodes = len(self.distNodes[0])

        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]

            if nameOfNode in self.values:
                nodes.append([self.values[nameOfNode]]*numNodes)

            if nameOfNode in self.dists:
                nodes.append(self.distNodes[orderdDistsNames.index(nameOfNode)])

        self.nodes = np.array(nodes)
        self.weights = np.array(self.weights)

        if self._performTransformation:
            # self.parameters = self.transformParameters(orderdDistsNames, self.nodes)
            self.parameters = Nodes.transformSamples(self.nodes, self.joinedStandardDists, self.joinedDists)
        else:
            self.parameters = self.nodes

        return self.nodes, self.weights, self.parameters

    def generateNodesFromListOfValues(self, fileName=None):
        nodes = []
        if fileName is not None and fileName:
            # reading a matrix of values from a parameters file
            raise NotImplementedError("Should have implemented this")
        else:
            # creating a matrix of values from default values in a config file
            for i in range(0, len(self.nodeNames)):
                nameOfNode = self.nodeNames[i]
                if nameOfNode in self.values:
                    nodes.append(self.values[nameOfNode])  # assumption self.values[nameOfNode] is a list
            # transpose for consistency, Nodes.nodes should be of size (stoch_dim x number_of_nodes)
            self.parameters = self.nodes = np.array(list(itertools.product(*nodes))).T

    def get_nodes_and_parameters(self):
        return self.nodes, self.parameters

    def transformParameters(self, orderdDistsNames, nodes):
        transformedNodes = np.array(nodes, copy=True)
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                transformedNodes[orderdDistsNames.index(nameOfNode)] = \
                    self.transformationFunctions[nameOfNode](transformedNodes[orderdDistsNames.index(nameOfNode)], \
                                                             self.transformationParameters[nameOfNode][0], \
                                                             self.transformationParameters[nameOfNode][1])
        return np.array(transformedNodes)

    @staticmethod
    def transformSamples(samples, distribution_r, distribution_q):
        """

        :param samples: array of samples from distribution_r
        :param distribution_r: 'standard' distribution
        :param distribution_q: 'user-defined' distribution
        :return: array of samples from distribution_q
        """
        return distribution_q.inv(distribution_r.fwd(samples))

    def __save__cpu_affinity(self):
        # Save cpu pinning: This is necessary, because through chaospy.generate_quadrature() -> scipy.linalg.eig_banded
        # the process is pinned to only one cpu (core)! That's why we have to save and reset it!!
        # This happens on the LRZ MAC-Cluster (snb and ati nodes) and on the LRZ Linux Cluster (CoolMUC2) on rank 0!
        self.cpu_affinity = psutil.Process().cpu_affinity()

    def __restore__cpu_affinity(self):
        if self.cpu_affinity != psutil.Process().cpu_affinity():
            psutil.Process().cpu_affinity(self.cpu_affinity)

    def printNodesSetup(self):
        self.assertConfiguration()

        nodesSetupTable = []

        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]

            if nameOfNode in self.values:
                nodesSetupTable.append([nameOfNode, self.values[nameOfNode]])

            if nameOfNode in self.dists:
                nodesSetupTable.append([nameOfNode, self.dists[nameOfNode]])

        return tabulate(nodesSetupTable, headers=["parameter", "value/dist"])

    def printNodes(self):
        resultTable = []
        #resultTable.append(self.nodeNames)

        nodes = self.nodes.T
        for i in range(0, len(nodes)):
            resultTable.append(nodes[i])

        str = tabulate(resultTable, headers=self.nodeNames, floatfmt="f") + "\n"
        str += "\n" + "{} - length of the Nodes array".format(len(nodes))
        return str

    def plotDistsSetup(self, fileName, numCollocationPointsPerDim, rule="G", show=False):

        #figure setup
        figure = plotter.figure(1, figsize=(6.5, 5))
        figure.canvas.set_window_title('simuluation node setup')

        dists = self.dists
        counter = 1
        numDists = len(dists)
        for distributionName in dists:
            #generate nodes and weights
            self.__save__cpu_affinity()
            nodes, weights = cp.generate_quadrature(numCollocationPointsPerDim,
                                                    dists[distributionName],
                                                    rule=rule)
            self.__restore__cpu_affinity()
            nodes = nodes[0]

            #plot quadrature nodes and weights
            plotter.subplot(numDists, 1, counter)
            plotter.title(distributionName + ' parameter\nnodes and weights\n'+ str(dists[distributionName]))
            plotter.plot(nodes, weights, 'bx-', label='weights')
            plotter.plot(nodes, np.zeros(nodes.size), 'r*', label='nodes')
            plotter.xlabel('node')
            plotter.ylabel('weight')
            plotter.legend() #enable the legend
            plotter.grid(True)

            counter = counter + 1

        plotter.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        #save figure
        plotter.savefig(fileName, format='pdf')

        #show figure
        if show:
            plotter.show() #show the plot

        plotter.close()

    def plotDists(self, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):

        fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)

        #prepare data
        sampled_data = self.joinedDists.sample(10**4)

        for sample_name, samples in zip(["sampled_data", "distNodes"], [sampled_data, self.distNodes]):
            orderdDistsNames = []
            for i in range(0, len(self.nodeNames)):
                nameOfNode = self.nodeNames[i]
                if nameOfNode in self.dists:
                    orderdDistsNames.append(nameOfNode)

            data_dict = {}
            for name, sample in zip(orderdDistsNames, samples):
                data_dict[name] = sample

            dataset = pd.DataFrame(data_dict)

            #plot
            figure = plotter.figure(1, figsize=(12, 4))
            sns.set()

            fontsize = 15
            plotter.rc('font', family='serif', size=fontsize)

            g = sns.pairplot(dataset)
            g.map_lower(sns.kdeplot)
            g.map_upper(sns.kdeplot)
            #g.map_diag(sns.kdeplot, lw=3)

            # Plotter settings
            #plotter.subplots_adjust(wspace=0.15, hspace=0.2, bottom=0.25, top=0.92, left=0.07, right=0.98)

            # save figure qoi_dist
            pdfFileName = fileName + "_" + "dists_pairplot_" + sample_name + '.pdf'
            pngFileName = fileName + "_" + "dists_pairplot_" + sample_name + '.png'
            plotter.savefig(pdfFileName, format='pdf')
            plotter.savefig(pngFileName, format='png')

            if display:
                plotter.show()

            plotter.close()

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

    def saveToFile(self, fileName):
        # save state file
        nodesFileName = fileName + '.simnodes.zip'
        with open(nodesFileName, 'wb') as f:
            #dill.dump(self, f)
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)
            #pickle._dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            # cPickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
            #dill.dump(self.nodes, f)

        #if self._performTransformation and self.parameters is not None:
        #    paramsFileName = fileName + '.simparams.pkl'
        #    with open(paramsFileName, 'wb') as f:
        #        dill.dump(self.parameters, f)


    # def saveToFile(self, fileName):
    #     jsonData = json.loads('{}')
    #     jsonData["nodeNames"] = self.nodeNames
    #     jsonData["values"] = self.values
    #     #jsonData["dists"] = self.dists
    #     #jsonData["joinedDists"] = self.joinedDists
    #     jsonData["distNodes"] = self.distNodes.tolist()
    #     jsonData["weights"] = self.weights.tolist()
    #     jsonData["nodes"] = self.nodes.tolist()
    #     jsonData["numSamplesOrScDim"] = self.numSamplesOrScDim
    #
    #     jsonFile = open(fileName, 'w')
    #     json.dump(jsonData, jsonFile, sort_keys=True)
    #     jsonFile.close()
    #
    # def loadFromFile(self, fileName):
    #     jsonFile = open(fileName, "r")
    #     jsonData = json.load(jsonFile)
    #     jsonFile.close()
    #
    #     self.nodeNames = jsonData["nodeNames"]
    #     self.values = jsonData["values"]
    #     #self.dists = jsonData["dists"]
    #     #self.joinedDists = jsonData["joinedDists"]
    #     self.distNodes = jsonData["distNodes"]
    #     self.weights = np.array(jsonData["weights"])
    #     self.nodes = np.array(jsonData["nodes"])
    #     self.numSamplesOrScDim = np.array(jsonData["numSamplesOrScDim"])

# if __name__ == "__main__":
#     #simNodes = SimulationNodes(['a', 'b'])
#     simNodes = SimulationNodes(['a', 'b'])
#     simNodes.setDist("a", cp.Normal(0, 1))
#     #simNodes.setValue("a", 3)
#     simNodes.setValue("b", 0.6)
#     #simNodes.setDist("a", cp.Normal(2, 3))
#
#     simNodes.printNodesSetup()
#     print "original" + str(simNodes.generateNodesForMC(5))
#     #print str(simNodes.generateNodesForSC(5))
#
#     simNodes.saveToFile("simNodesOut.txt")
#
#     simNodes2 = SimulationNodes([])
#     simNodes2.loadFromFile("simNodesOut.txt")
#     print "loaded" + str(simNodes2.generateNodesForMC(5))
#
#
#     pass
