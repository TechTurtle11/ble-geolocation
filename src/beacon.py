from httplib2 import RETRIES
from importlib_metadata import re
import numpy as np
import sklearn.gaussian_process as gp
from constants import PROPAGATION_CONSTANT


class Beacon():


    def __init__(self,address,training_data,position=None,rssi_at_position=None) -> None:
        self._mac_address = address
        self._training_data = training_data
        self._map = self.create_beacon_map(training_data)

        Y = training_data.T[0]
        X = training_data.T[1:].T

        # test assuming actual position is in training data 
        index = np.argmax(Y)
        self._position = X[index]
        self._rssi_at_position = Y[index]





    def create_beacon_map(self,training_data):
        Y = training_data.T[0]
        X = training_data.T[1:].T

        kernel = gp.kernels.RBF(length_scale=5)
        gpr = gp.GaussianProcessRegressor(
            kernel=kernel, random_state=0, normalize_y=False).fit(X,Y)
        
        return gpr

    def predict_offset_rssi(self,point):
        return -10*PROPAGATION_CONSTANT*np.log10(np.linalg.norm(point-self._position)) + self._rssi_at_position


    def predict_rssi(self,point,offset = False):

        predicted_rssi, cell_cov = self._map.predict([point], return_cov=True)

        if cell_cov[0] > 0.3 and offset: # for sparse areas
            predicted_rssi = self.predict_offset_rssi(point)
        else:
            predicted_rssi = predicted_rssi[0] # unpacking

        return predicted_rssi


    @property
    def get_map(self):
        return self._map

    @property
    def position(self):
        return self._position
    
    @property
    def training_data(self):
        return self._training_data

    
    def __str__(self) -> str:
        return self._mac_address



def create_beacons(training_data: dict):
    maps = {}
    for beacon, observations in training_data.items():
        maps[beacon] = Beacon(beacon,observations)

    return maps