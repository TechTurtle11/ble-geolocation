from enum import Enum

PROPAGATION_CONSTANT = 3.5

class Prior(Enum):
    UNIFORM = 1
    LOCAL = 2

class MapAttribute(Enum):
    PROB = 1
    COV = 2