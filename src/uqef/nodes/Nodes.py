"""
Created on 10.05.2015

@author: flo
"""

import chaospy as cp
import itertools
import inspect
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

        # intermediate values if reading nodes and weight from some file
        self.nodes_read_from_file = None
        self.weights_read_from_file = None

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

    def set_joined_dists(self):
        orderdDists = []
        orderdDistsNames = []
        orderdStandardDists = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
                if self._performTransformation:
                    orderdStandardDists.append(self.standardDists[nameOfNode])
        if len(self.dists) > 0 and orderdDists:
            self.joinedDists = cp.J(*orderdDists)
            if self._performTransformation and orderdStandardDists:
                self.joinedStandardDists = cp.J(*orderdStandardDists)
            else:
                self.joinedStandardDists = None
        else:
            self.joinedDists = None
            self.joinedStandardDists = None

    def generateNodesForMC(self, numSamples, rule="R", read_nodes_from_file=False, parameters_file_name=None,
                           parameters_setup_file_name=None):
        if self.numSamplesOrScDim == numSamples:
            return self.nodes, self.parameters

        self.assertConfiguration()
        self.numSamplesOrScDim = numSamples

        #order the distributes to get a defined order
        orderdDists = []
        orderdDistsNames = []
        orderdStandardDists = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
                if self._performTransformation:
                    orderdStandardDists.append(self.standardDists[nameOfNode])

        stochastic_dim = len(orderdDists)  # len(list(self.dists.keys()))

        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            if self._performTransformation:
                self.joinedStandardDists = cp.J(*orderdStandardDists)

            if read_nodes_from_file and parameters_file_name is not None:
                nodes_and_weights_array = np.loadtxt(parameters_file_name, delimiter=',')
                self.nodes_read_from_file = nodes_and_weights_array[:, :stochastic_dim].T
                self.weights_read_from_file = nodes_and_weights_array[:, stochastic_dim]
                # transform nodes and weight you have read from the file
                self.distNodes, self.weights = self._transform_nodes_and_weights_read_from_file(
                    nodes_read_from_file=self.nodes_read_from_file, 
                    weights_read_from_file=self.weights_read_from_file,
                    performTransformation=self._performTransformation, 
                    stochastic_dim=stochastic_dim, 
                    parameters_setup_file_name=parameters_setup_file_name
                    )
                self.numSamplesOrScDim = len(self.distNodes[0])
            else:
                if self._performTransformation:
                    self.distNodes = self.joinedStandardDists.sample(size=numSamples, rule=rule)#.round(4)
                else:
                    #self.distNodes = self.joinedDists.sample(numSamples, rule=rule).round(4)
                    #self.distNodes = cp.generate_samples(order=numSamples, domain=self.joinedDists, rule=rule).round(4)
                    self.distNodes = self.joinedDists.sample(size=numSamples, rule=rule)#.round(4)

        nodes = []

        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]

            if nameOfNode in self.values:
                nodes.append([self.values[nameOfNode]]*numSamples)

            if nameOfNode in self.dists:
                if len(self.dists) == 1:
                    nodes.append(self.distNodes)
                else:
                    nodes.append(self.distNodes[orderdDistsNames.index(nameOfNode)])

        self.nodes = np.array(nodes)
        self.weights = np.array(self.weights)  # MC has no weights, but after generation, we want an array

        if self._performTransformation:
            # self.parameters = self.transformParameters(orderdDistsNames, self.nodes)
            self.parameters = Nodes.transformSamples(self.nodes, self.joinedStandardDists, self.joinedDists)
        else:
            self.parameters = self.nodes

        return self.nodes, self.parameters

    def generateNodesForSC(self, numCollocationPointsPerDim, rule="G", sparse=False, read_nodes_from_file=False,
                           parameters_file_name=None, parameters_setup_file_name=None):

        if self.numSamplesOrScDim == numCollocationPointsPerDim:
            return self.nodes, self.weights, self.parameters

        self.numSamplesOrScDim = numCollocationPointsPerDim

        orderdDists = []
        orderdDistsNames = []
        orderdStandardDists = []
        # self.joinedDists = []
        # self.distNodes = []
        # self.weights = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
                if self._performTransformation:
                    orderdStandardDists.append(self.standardDists[nameOfNode])

        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            self.__save__cpu_affinity()
            growth = True if (rule == "c" and not sparse) else False  # according to: https://github.com/jonathf/chaospy/issues/139

            if self._performTransformation:
                self.joinedStandardDists = cp.J(*orderdStandardDists)

            stochastic_dim = len(orderdDists)  # len(list(self.dists.keys()))

            if read_nodes_from_file and parameters_file_name is not None:
                nodes_and_weights_array = np.loadtxt(parameters_file_name, delimiter=',')
                self.nodes_read_from_file = nodes_and_weights_array[:, :stochastic_dim].T
                self.weights_read_from_file = nodes_and_weights_array[:, stochastic_dim]
                # transform nodes and weight you have read from the file
                self.distNodes, self.weights = self._transform_nodes_and_weights_read_from_file(
                    nodes_read_from_file=self.nodes_read_from_file, 
                    weights_read_from_file=self.weights_read_from_file,
                    performTransformation=self._performTransformation, 
                    stochastic_dim=stochastic_dim, 
                    parameters_setup_file_name=parameters_setup_file_name
                    )
                # TODO Update self.numSamplesOrScDim based on what is read from file!
                self.numSamplesOrScDim = numCollocationPointsPerDim
            else:
                if self._performTransformation:
                    dist_for_quadrature = self.joinedStandardDists
                else:
                    dist_for_quadrature = self.joinedDists
                self.distNodes, self.weights = cp.generate_quadrature(
                    numCollocationPointsPerDim, dist_for_quadrature, rule=rule, growth=growth, sparse=sparse)

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
            self.parameters = Nodes.transformSamples(self.nodes, self.joinedStandardDists, self.joinedDists)
        else:
            self.parameters = self.nodes

        return self.nodes, self.weights, self.parameters

    def generateNodesFromListOfValues(self, read_nodes_from_file=False,
                                      parameters_file_name=None,
                                      parameters_setup_file_name=None):
        nodes = []
        if read_nodes_from_file and parameters_file_name is not None and parameters_file_name:
            # reading a matrix of values from a parameters file; read_nodes_from_file=True?
            raise NotImplementedError("Should have implemented this")
        else:
            # creating a matrix of values from default values in a config file
            for i in range(0, len(self.nodeNames)):
                nameOfNode = self.nodeNames[i]
                if nameOfNode in self.values:
                    values = self.values[nameOfNode]
                    if not isinstance(values, list):
                        values = [values, ]
                    nodes.append(values)  # assumption self.values[nameOfNode] is a list
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
    def transformSamples_lin_or_nonlin(samples, distribution_r, distribution_q, linear=True):
        """
        Note: Linear transformation only covers the case of transforming r.v. distributed accoring to some
         Uniform distribution to standard Uniform distribution (U[-1,1] or U[0,1])
        """
        if linear:
            dim = len(distribution_r)
            assert len(distribution_r) == len(distribution_q)
            _a = np.empty([dim, 1])
            _b = np.empty([dim, 1])

            for i in range(dim):
                r_lower = distribution_r[i].lower
                r_upper = distribution_r[i].upper
                q_lower = distribution_q[i].lower
                q_upper = distribution_q[i].upper

                if r_lower == -1:
                    _a[i] = (q_lower + q_upper) / 2
                    _b[i] = (q_upper - q_lower) / 2
                elif r_lower == 0:
                    _a[i] = q_lower
                    _b[i] = (q_upper - q_lower)

            return _a + _b * samples
        else:
            return distribution_q.inv(distribution_r.fwd(samples))

    @staticmethod
    def transformSamples(samples, distribution_r, distribution_q):
        """
        https://chaospy.readthedocs.io/en/master/user_guide/advanced_topics/generalized_polynomial_chaos.html
        :param samples: array of samples from distribution_r
        :param distribution_r: 'standard' distribution
        :param distribution_q: 'user-defined' distribution
        :return: array of samples from distribution_q
        """
        # TODO Think if transformation should be done dimension wise
        return distribution_q.inv(distribution_r.fwd(samples))

    @staticmethod
    def transformSamples_from_uniform(samples, distribution_r, distribution_q):
        """
        :param samples: array of samples from distribution_r, when distribution_r is U[-1,1] or U[0,1]
        :param distribution_r: 'standard' distribution either U[-1,1] or U[0,1]
        :param distribution_q: 'user-defined' distribution
        :return: array of samples from distribution_q
        """
        dim = len(distribution_r)
        assert len(distribution_r) == len(distribution_q)
        _a = np.empty([dim, 1])
        _b = np.empty([dim, 1])

        for i in range(dim):
            r_lower = distribution_r[i].lower
            q_lower = distribution_q[i].lower
            q_upper = distribution_q[i].upper

            if r_lower == -1:
                _a[i] = (q_lower + q_upper) / 2
                _b[i] = (q_upper - q_lower) / 2
            elif r_lower == 0:
                _a[i] = q_lower
                _b[i] = (q_upper - q_lower)

        return _a + _b * samples

    def _transform_nodes_and_weights_read_from_file(
        self, nodes_read_from_file, weights_read_from_file, 
        performTransformation, stochastic_dim, parameters_setup_file_name=None):
        """
        Important function when reading position of the nodes from some file. Ensure that these nodes are distributed
        according to desired distribution specified in the configuration file.

        :nodes_read_from_file: Nodes read from file
        :param performTransformation: If True, then the nodes are transformed to the desired 'standard' distribution
        and just later on the parameters used to stimulate the model are transformed to the desired 'user-defined' distribution
        :param stochastic_dim:
        :param parameters_setup_file_name: From this file we read a setup/distribution according to which the read nodes are distributed
        :return:
        """
        distsOfNodesFromFile = []
        if parameters_setup_file_name is not None:
            with open(parameters_setup_file_name) as f:
                parameters_configuration_object = json.load(f)
            for parameter_config in parameters_configuration_object["parameters"]:
                # node values and distributions -> automatically maps dists and their parameters by reflection mechanisms
                cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
                dist_parameters_values = []
                for p in cp_dist_signature.parameters:
                    dist_parameters_values.append(parameter_config[p])
                distsOfNodesFromFile.append(getattr(cp, parameter_config["distribution"])(*dist_parameters_values))
        else:
            # Hard-coded by default, assumption is that read nodes have Uniform(0,1) distribution
            for _ in range(stochastic_dim):
                distsOfNodesFromFile.append(cp.Uniform())
                # distsOfNodesFromFile.append(cp.Uniform(lower=0.0, upper=1.0))

        jointDistOfNodesFromFile = cp.J(*distsOfNodesFromFile)

        if performTransformation:
            distNodes = Nodes.transformSamples(
                nodes_read_from_file, jointDistOfNodesFromFile, self.joinedStandardDists)
            weights = weights_read_from_file
        else:
            distNodes = Nodes.transformSamples(
                nodes_read_from_file, jointDistOfNodesFromFile, self.joinedDists)
            weights = weights_read_from_file
        return distNodes, weights

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
        figure.canvas.manager.set_window_title('simuluation node setup')

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
