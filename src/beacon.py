import sklearn.gaussian_process as gp
from sklearn.linear_model import LinearRegression
from constants import beacon_locations



class Beacon():


    def __init__(self,address,training_data,) -> None:
        self._mac_address = address
        self._training_data = training_data
        self._map = self.create_beacon_map(training_data)
        self._gpr,self._lg = self.create_beacon_map(training_data)

        Y = training_data.T[0]
        X = training_data.T[1:].T


        self._position = beacon_locations[address]





    def create_beacon_map(self,training_data):
        Y = training_data.T[0]
        X = training_data.T[1:].T

        kernel =gp.kernels.RBF(length_scale=5,length_scale_bounds=(0.00001,100))
        gpr = gp.GaussianProcessRegressor(
            kernel=kernel, random_state=0, normalize_y=False,n_restarts_optimizer=2).fit(X,Y)
            
        lg = LinearRegression().fit(X,Y)
        return gpr,lg

    def predict_offset_rssi(self,point):
        return self._lg.predict([point])


    def predict_rssi(self,point,offset = False):

        predicted_rssi, cell_cov = self._gpr.predict([point], return_cov=True)

        if cell_cov[0] > 0.5 and offset: # for sparse areas
            predicted_rssi = self.predict_offset_rssi(point)[0]
        else:
            predicted_rssi = predicted_rssi[0] # unpacking

        return predicted_rssi


    @property
    def get_map(self):
        return self._gpr

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