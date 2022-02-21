from enum import Enum

PROPAGATION_CONSTANT = 3.5
beacon_locations = {'e4:5f:01:63:71:64': [6, 3], 'e4:5f:01:63:71:e5': [-2.4, 12.6],
                    'e4:5f:01:63:71:55': [-2.4, 6], 'e4:5f:01:63:71:b5': [3, 9.6], 'e4:5f:01:63:71:a3': [0, 0]}

BEACON_MAC_ADDRESSES = ['e4:5f:01:63:71:64','e4:5f:01:63:71:e5','e4:5f:01:63:71:55', 'e4:5f:01:63:71:b5', 'e4:5f:01:63:71:a3']
BEACON_WINDOW = 1  # Seconds
BEACON_SAMPLES_PER_WINDOW = 10


class Prior(Enum):
    UNIFORM = 1
    LOCAL = 2


class MapAttribute(Enum):
    PROB = 1
    COV = 2


class MeasurementProcess(Enum):
    ALL = 1
    MEAN = 2
    QUANTILE = 3
    MEDIAN = 4