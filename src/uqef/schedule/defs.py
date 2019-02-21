"""
Scheduling enum definitions

@author: Florian Kuenzner
"""

from enum import Enum


class Type(Enum):
    """
    Type describes ho the work items are organised
     - WORK_LIST   : the work items are organised as a "simple" list
     - WORK_PACKAGE: the work items are organised as packages ("for each worker one package")
    """
    WORK_LIST    = 1
    WORK_PACKAGE = 2


Types = {
    "WORK_LIST"    : Type.WORK_LIST
   ,"WORK_PACKAGE" : Type.WORK_PACKAGE
}


class Algorithm(Enum):
    """
    Algorithm is the type of algorithm to schedule the work
    """
    FCFS     = 1
    LPT      = 2
    SPT      = 3
    MULTIFIT = 4


Algorithms = {
    "FCFS"     : Algorithm.FCFS
   ,"LPT"      : Algorithm.LPT
   ,"SPT"      : Algorithm.SPT
   ,"MULTIFIT" : Algorithm.MULTIFIT
}


class Strategy(Enum):
    """
    Strategy is the strategy how the schedule algorithm works
     - FIXED_ALTERNATE: Takes one item after another and places it into the next work package
     - FIXED_LINEAR   : Takes one item after another and places it into the first bin, after it is
                        full, it proceeds with the next work package
     - DYNAMIC        : Give the next work item to the next free worker
    """
    FIXED_ALTERNATE = 1
    FIXED_LINEAR    = 2
    DYNAMIC         = 3


Strategies = {
    "FIXED_ALTERNATE" : Strategy.FIXED_ALTERNATE
   ,"FIXED_LINEAR"    : Strategy.FIXED_LINEAR
   ,"DYNAMIC"         : Strategy.DYNAMIC
}