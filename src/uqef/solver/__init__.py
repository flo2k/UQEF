"""
Solver package

@author: Florian Kuenzner
"""

# abstract solver
from .Solver import Solver

# solver implementations
from .LinearSolver import LinearSolver
from .ParallelSolver import ParallelSolver
from .MpiPoolSolver import MpiPoolSolver
from .MpiSolver import MpiSolver

# solver times
from .SolverTimes import *
