from abc import ABC
from itertools import chain
import math
from pathlib import Path

import numpy as np
from beacon import create_beacons
from localisation import predict_positions
from map import Map

from measurement import process_training_data
import file_helper as fh
import general_helper as gh
import constants as const


class BaseModel(ABC):

    def predict_position(self, rssi_measurement):
        pass

    def predict_convergent_position(self, rssi_measurement, reset_map):
        return self.predict_position(rssi_measurement)


class GaussianProcessModel(BaseModel):
    """
    Implementation of gaussian model:
    This uses the gaussian process to produce the cell map then co-locates to the cell point
    with the highest probabilty which passes the standard deviation test,
    """

    def __init__(self, training_data_filepath: Path, prior, bottom_corner=None, shape=None, cell_size=1):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        training_data = process_training_data(
            training_data, type=const.MeasurementProcess.MEDIAN)
        self.beacons = create_beacons(self.beacon_positions, training_data)
        if bottom_corner is None or shape is None:
            bottom_corner, shape = self.get_map_dimensions(
                self.beacon_positions, training_data, cell_size)
        self.area_map = Map(bottom_corner, shape, cell_size)
        self.prior = prior

    def predict_position(self, rssi_measurement):
        """
        Predicts the position of the target device
        """

        calculated_cells = self.area_map.calculate_cell_probabilities(
            rssi_measurement, self.beacons, self.prior)
        calculated_cells = np.array(
            [cell for cell in calculated_cells if cell.std < 0.4])
        sorted_cells = sorted(
            calculated_cells, key=lambda c: c.probability, reverse=False)
        self.area_map.previous_cell = sorted_cells[0]

        return sorted_cells[0].center

    def predict_convergent_position(self, rssi_measurement, reset_map):
        if reset_map:
            self.area_map.reset_map()

        return self.predict_position(rssi_measurement)

    def get_map_dimensions(self, beacon_positions, training_data, cell_size):
        """
        Obtains map dimensions from the training data
        """

        positions = np.array(
            list(chain.from_iterable(training_data.values())))[:, 1:]

        positions = np.append(positions, list(
            beacon_positions.values()), axis=0)

        bottom_corner = np.array(
            [np.amin(positions[:, 0]), np.amin(positions[:, 1])])
        top_corner = np.array(
            [np.amax(positions[:, 0]), np.amax(positions[:, 1])])

        shape = np.ceil((top_corner-bottom_corner) / cell_size)

        return bottom_corner, shape


class GaussianKNNModel(GaussianProcessModel):
    """
    Implementation of gaussian model with k-nearest neighbours on the area map:
    This uses the gaussian process to produce the cell map then uses the 3 lowest cell
    centers which pass the covariance condition have high probabilities and produces
    a weighted mean of there corresponding positions.
    """

    def predict_position(self, rssi_measurement):
        """
        Predicts the position of the target device
        """

        calculated_cells = self.area_map.calculate_cell_probabilities(
            rssi_measurement, self.beacons, self.prior)

        calculated_cells = np.array(
            [cell for cell in calculated_cells if cell.std < 0.4])
        sorted_cells = sorted(
            calculated_cells, key=lambda c: c.probability, reverse=False)

        self.area_map.previous_cell = sorted_cells[0]
        k = 3
        first_k = sorted_cells[:k]

        position = np.zeros(2)

        prob_sum = sum([abs(cell.probability) for cell in first_k])
        for cell in first_k:
            position += (abs(cell.probability) / prob_sum) * cell.center

        return position


class WKNN(BaseModel):
    """
    Implementation of a weighted k-nearest neighbours model:
    The kNN algorithm uses k = 3 and comapares the input measurement to all training data measurment, then
    uses the three closest (lowest root mean square error(RMSE)) measurements and produces
    a weighted mean of there corresponding positions.
    """

    def __init__(self, training_data_filepath: Path):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        self.training_data = process_training_data(
            training_data, type=const.MeasurementProcess.MEDIAN)

    def predict_position(self, rssi_measurement, k=3):
        """
        Predicts the position of the target device
        """
        distances = {}
        for beacon, data in self.training_data.items():

            for line in data:
                d_hash = gh.hash_2D_coordinate(*line[1:])
                if not d_hash in distances.keys():
                    distances[d_hash] = [np.array(line[1:]), 0, 1*10**-6] #(position,distance_scratch,beacons_used)
                if beacon in rssi_measurement.keys():
                    distances[d_hash][1] += np.square(
                        line[0] - rssi_measurement[beacon])
                    distances[d_hash][2] += 1

        for h, data in distances.items():
            distances[h][1] = np.sqrt(data[1]/distances[h][2])

        sorted_points = sorted(list(distances.values()), key=lambda p: p[1])
        first_k = np.array(sorted_points[:k])

        position = np.zeros(2)

        distance_sum = sum(
            [1 / distance for _, distance, _ in first_k]) + 1*10**-9
        for point, distance, _ in first_k:
            if distance == 0:
                return point
            else:
                position += ((1/distance) / distance_sum) * point

        return position


class KNN(BaseModel):
    """
    Implementation of a k-nearest neighbours model:
    The kNN algorithm uses k = 3 and takes the three strongest signals and produces
    a weighted mean of the corresponding beacon positions.
    """

    def __init__(self, training_data_filepath: Path):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)

    def predict_position(self, rssi_measurement, k=3):
        """
        Predicts the position of the target device
        """
        sorted_points = sorted(
            list(rssi_measurement.items()), key=lambda p: p[1], reverse=True)
        first_k = sorted_points[:k]

        position = np.zeros(2)

        rssi_sum = sum([abs(1 / rssi) for _, rssi in first_k])

        for beacon, rssi in first_k:
            position += (abs(1/rssi) / rssi_sum) * \
                self.beacon_positions[beacon]

        return position


class PropagationModel(BaseModel):
    """
    Implementation of a propogation model:
    The propogation model, uses a linear path loss model to predict distances from beacons
    and co-locates the user with them.
    """

    def __init__(self, training_data_filepath: Path, n):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        training_data = process_training_data(
            training_data, type=const.MeasurementProcess.MEAN)

        # get constant values closest to 1
        beacon_constants = {}
        for beacon, data in training_data.items():
            beacon_position = self.beacon_positions[beacon]
            distances = np.linalg.norm(data[:, 1:]-beacon_position, axis=1) - 1
            closest_index = np.argmin(distances)
            beacon_constants[beacon] = (
                data[closest_index, 0], 1+distances[closest_index])

        self.distance_functions = {}
        for beacon, constants in beacon_constants.items():
            self.distance_functions[beacon] = lambda rssi: constants[1] * \
                np.power((rssi-constants[0])/-10*n, 10)

    def predict_position(self, rssi_measurement):
        """
        Predicts the position of the target device
        """

        distance_sum = 1*10**-6
        beacon_distances = {}
        for beacon, measurement in rssi_measurement.items():
            distance = self.distance_functions[beacon](measurement)
            distance_sum += distance
            beacon_distances[beacon] = distance

        position = np.zeros(2)
        distance_sum = sum(
            [1 / distance for _, distance in beacon_distances.items()]) + 1*10**-9
        for beacon, distance in beacon_distances.items():
            if distance == 0:
                return self.beacon_positions[beacon]
            else:
                position += ((1/distance) / distance_sum) * \
                    self.beacon_positions[beacon]

        return position


class ProximityModel(BaseModel):
    """
    Implementation of a proximity model:
    The proximity algorithm selects the beacon with the strongest received signal at
    a given time and co-locates the user with it.
    """

    def __init__(self, training_data_filepath: Path) -> None:
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)

    def predict_position(self, rssi_measurement):
        """
        Predicts the position of the target device
        """

        max_beacon = None
        max_rssi = -np.inf
        for beacon, measurement in rssi_measurement.items():
            if measurement > max_rssi:
                max_beacon = beacon
                max_rssi = measurement

        return self.beacon_positions[max_beacon]
