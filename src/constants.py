from enum import Enum

PROPAGATION_CONSTANT = 3.5

BEACON_MAC_ADDRESSES = ['e4:5f:01:63:71:64', 'e4:5f:01:63:71:e5',
                        'e4:5f:01:63:71:55', 'e4:5f:01:63:71:b5', 'e4:5f:01:63:71:a3']
BEACON_WINDOW = 1  # Seconds
BEACON_SAMPLES_PER_WINDOW = 10


class Prior(Enum):
    """ Indicates how the prior should be calculated in gaussian
    """
    UNIFORM = 1
    LOCAL = 2


class MapAttribute(Enum):
    """ Used in plotting to indicate type of heatmap
    """
    PROB = 1
    COV = 2


class MeasurementProcess(Enum):
    """ Indicates how rssi measurement windows data is turned into training data.
    """
    ALL = 1
    MEAN = 2
    QUANTILE = 3
    MEDIAN = 4


class Model(Enum):
    """ Indicates indoor positioning model
    """

    GAUSSIAN = "Gaussian"
    GAUSSIANKNN = "Gaussian KNN"
    GAUSSIANMINMAX = "Gaussian MinMax"
    KNN = "KNN"
    WKNN = "WKNN"
    PROPOGATION = "Propagation"
    PROXIMITY = "Proximity"
