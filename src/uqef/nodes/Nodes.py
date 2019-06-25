"""
Created on 10.05.2015

@author: flo
"""

import chaospy as cp
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plotter
import json
import dill
import psutil


class Nodes(object):
    """
    Nodes represents the nodes and parameters for a UQ simulation
    """

    @staticmethod
    def restoreFromFile(fileName):
        with open(fileName, 'rb') as f:
            return dill.load(f)

    def __init__(self, nodeNames):
        self.nodeNames = nodeNames
        self.values = {}
        self.dists = {}
        self.joinedDists=[]
        self.distNodes=[]
        self.weights=[]
        self.nodes=[]
        self.numSamplesOrScDim = None
        
    def setValue(self, nodeName, value):
        self.assertNodeName(nodeName)
        
        self.values[nodeName] = value
    
    def setDist(self, nodeName, dist):
        self.assertNodeName(nodeName)
        
        self.dists[nodeName] = dist
    
    def assertNodeName(self, nodeName):
        assert nodeName in self.nodeNames, "name of node " + nodeName + " not registered."
    
    def assertConfiguration(self):
        numRegisteredNodes = len(self.nodeNames)
        numValues = len(self.values)
        numDists = len(self.dists)
        
        assert (numValues + numDists) == numRegisteredNodes, "not enough values registered"
    
    def generateNodesForMC(self, numSamples):
        if self.numSamplesOrScDim == numSamples:
            return self.nodes

        self.assertConfiguration()
        self.numSamplesOrScDim = numSamples
        
        #order the distributes to get a defined order
        orderdDists = []
        orderdDistsNames = []
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
        
        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            distNodes = self.joinedDists.sample(numSamples)
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
        self.weights = np.array(self.weights) #MC has no weights, but after generation, we want a array
        return self.nodes
    
    def generateNodesForSC(self, numCollocationPointsPerDim, rule="G", sparse=False):

        if self.numSamplesOrScDim == numCollocationPointsPerDim:
            return self.nodes, self.weights

        self.numSamplesOrScDim = numCollocationPointsPerDim

        orderdDists = []
        orderdDistsNames = []
        self.joinedDists=[]
        self.distNodes=[]
        self.weights=[]
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            if nameOfNode in self.dists:
                orderdDists.append(self.dists[nameOfNode])
                orderdDistsNames.append(nameOfNode)
        
        if len(self.dists) > 0:
            self.joinedDists = cp.J(*orderdDists)
            self.__save__cpu_affinity()
            growth = True if (rule == "c" and sparse == False) else False  # according to: https://github.com/jonathf/chaospy/issues/139
            self.distNodes, self.weights = cp.generate_quadrature(numCollocationPointsPerDim, 
                                                                  self.joinedDists, 
                                                                  rule=rule,
                                                                  growth=growth,
                                                                  sparse=sparse)
            self.__restore__cpu_affinity()
        
        nodes = []
        if len(self.distNodes) == 0:
            numNodes=numCollocationPointsPerDim
        else:
            numNodes=len(self.distNodes[0])
        
        for i in range(0, len(self.nodeNames)):
            nameOfNode = self.nodeNames[i]
            
            if nameOfNode in self.values:
                nodes.append([self.values[nameOfNode]]*numNodes)
                
            if nameOfNode in self.dists:
                nodes.append(self.distNodes[orderdDistsNames.index(nameOfNode)])

        self.nodes = np.array(nodes)
        self.weights = np.array(self.weights)
        return self.nodes, self.weights

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
        str += "\n" + "{} black-box models runs required".format(len(nodes))
        return str
    
    def plotDistsSetup(self, fileName, numCollocationPointsPerDim, rule="G", show=False):
        
        #figure setup
        figure = plotter.figure(1, figsize=(6.5, 5))
        figure.canvas.set_window_title('simuluation node setup')

        dists=self.dists
        counter = 1
        numDists = len(dists)
        for distributionName in dists:
            #generate nodes and weights
            self.__save__cpu_affinity()
            nodes, weights = cp.generate_quadrature(numCollocationPointsPerDim, 
                                                    dists[distributionName], 
                                                    rule=rule)
            self.__restore__cpu_affinity()
            nodes=nodes[0]
             
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

    def saveToFile(self, fileName):
        # save state file
        statFileName = fileName + '.simnodes'
        with open(statFileName, 'wb') as f:
            #pickle.dump(list(self), f)
            dill.dump(self, f)

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