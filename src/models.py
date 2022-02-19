from abc import ABC
import math
from pathlib import Path

import numpy as np
from beacon import create_beacons
from map import Map

from measurement import process_training_data
import file_helper as fh
import general_helper as gh
import constants as const


class BaseModel(ABC):


    def predict_position(self, rssi_measurement):
        pass



class GaussianProcessModel(BaseModel):

    def __init__(self, training_data_filepath: Path, prior,starting_point = [-3,-3],ending_point = [10,17],cell_size=1):
        beacon_positions, training_data = fh.load_training_data(training_data_filepath, windows=True)
        training_data = process_training_data(training_data)
        self.beacons = create_beacons(beacon_positions, training_data)
        self.area_map = Map(starting_point, ending_point, cell_size)
        self.prior = prior

    def predict_position(self, rssi_measurement,previous_cell = None):

        calculated_cells = self.area_map.calculate_cell_probabilities(rssi_measurement,self.beacons,previous_cell,self.prior)

        sorted_cells = sorted(calculated_cells, key=lambda c: c.probability, reverse=False)



        return sorted_cells[0].center


class GaussianKNNModel(GaussianProcessModel):



    def predict_position(self, rssi_measurement,previous_cell = None):

        calculated_cells = self.area_map.calculate_cell_probabilities(rssi_measurement,self.beacons,previous_cell,self.prior)

        sorted_cells = sorted(calculated_cells, key=lambda c: c.probability, reverse=False)

        k = 3
        first_k = sorted_cells[:k]

        position = np.zeros(2)

        prob_sum = sum([abs(cell.probability) for cell in first_k])
        for cell in first_k:
            position += (abs(cell.probability) / prob_sum) * cell.center


        return position


class KNN(BaseModel):

    def __init__(self, training_data_filepath:Path):
        self.beacon_positions, training_data = fh.load_training_data(training_data_filepath, windows=True)
        self.training_data = process_training_data(training_data,type = const.MeasurementProcess.MEAN)


    def predict_position(self,rssi_measurement, k=3):

        distances = {} 
        for beacon, data in self.training_data.items():
            
            for line in data:
                d_hash = gh.hash_2D_coordinate(*line[1:])
                if not d_hash in distances.keys():
                    distances[d_hash] = [np.array(line[1:]),0]
                distances[d_hash][1] +=  (np.square(line[0] - rssi_measurement[beacon]))**2


        for h, data in distances.items():
            distances[h][1] = np.sqrt(data[1]/len(self.beacon_positions))


        sorted_points = sorted(list(distances.values()), key = lambda p : p[1])
        first_k = sorted_points[:k]

        position = np.zeros(2)

        distance_sum = sum([distance for _, distance in first_k])
        for point, distance in first_k:
            position += (distance / distance_sum) * point

        return position
