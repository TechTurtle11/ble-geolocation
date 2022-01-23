from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression

from constants import PROPAGATION_CONSTANT


class Beacon():


    def __init__(self,address,training_data,position,rssi_at_position) -> None:
        self._mac_address = address
        self._training_data = training_data
        self._map = self.create_beacon_survey_map(training_data)
        self._position = position
        self._rssi_at_position = rssi_at_position





    def create_beacon_map(self,training_data):
        Y = training_data.T[0]
        X = training_data.T[1:].T

        kernel = RBF(length_scale=5)
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=0, normalize_y=False).fit(X,Y)
        
        return gpr

    def get_offset_rssi(self,point):
        return -10*PROPAGATION_CONSTANT*np.log10(np.linalg(point-self._position)) + self._rssi_at_position

    @property
    def get_map(self):
        return self._map

    
    def __str__(self) -> str:
        return self._mac_address