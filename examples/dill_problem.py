from enum import Enum
import dill

class Strategy(Enum):
    StrategyA = 1
    StrategyB = 2

s = Strategy.StrategyA

with open("s.dump", 'wb') as f:
    dill.dump(s, f)
