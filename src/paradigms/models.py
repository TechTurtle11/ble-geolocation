import abc
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
from paradigms.beacon import create_beacons
from paradigms.map import Map

from measurement import process_training_data
import utils.file_helper as fh
import utils.general_helper as gh
import utils.constants as const


class BaseModel(abc.ABC):
    """
    Abstract class for a model. All models are instances of, this implements general behaviour and methods
    """

    @abc.abstractmethod
    def predict_position(self, rssi_measurement: dict):
        pass

    def predict_convergent_position(self, rssi_measurement: dict, reset_map: bool):
        return self.predict_position(rssi_measurement)

    def weighted_centroid_localisation(self, points: list, values: list, func: Callable):
        """
        Implement weigthed centroid localisation, the weights are created by applying the function to each value in values.
        """

        weights = [func(value) for value in values]

        weights_sum = sum(weights)

        position = np.zeros(2)
        for point, weight in zip(points, weights):
            position += (weight / weights_sum) * point
        return position

    def bounding_box(self, points: list, values: list, func: Callable):
        """
        Implements the bounding box localisation method, weights are created by applying the function to each value in values.
        """

        weights = np.array([func(value) for value in values])

        change = np.repeat(weights.reshape(
            1, len(weights)), len(points[0]), axis=0).T

        x_add = points + change
        x_sub = points - change

        x_min = np.max(x_sub[:, 0])
        x_max = np.min(x_add[:, 0])
        y_min = np.max(x_sub[:, 1])
        y_max = np.min(x_add[:, 1])

        x_est = 0.5 * np.array([x_min + x_max, y_min + y_max])

        return x_est


class GaussianProcessModel(BaseModel):
    """
    Implementation of gaussian model:
    This uses the gaussian process to produce the cell map then co-locates to the cell point
    with the highest probabilty which passes the standard deviation test,
    """

    def __init__(
            self, training_data_filepath: Path, prior: const.Prior, bottom_corner: list = None,
            shape: list = None, cell_size: int = 1, filter: bool = False):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        training_data = process_training_data(
            training_data, type=const.MeasurementProcess.QUANTILE, filter=filter)
        self.beacons = create_beacons(self.beacon_positions, training_data)
        if bottom_corner is None or shape is None:
            bottom_corner, shape = self.get_map_dimensions(
                self.beacon_positions, training_data, cell_size)
        self.area_map = Map(bottom_corner, shape, cell_size)
        self.prior = prior

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """

        calculated_cells = self.area_map.calculate_cell_probabilities(
            rssi_measurement, self.beacons, self.prior)

        stds = np.array([cell.std for cell in calculated_cells])
        std_median = np.median(stds)
        calculated_cells = np.array(
            [cell for cell in calculated_cells if cell.std < std_median + np.std(stds)])
        sorted_cells = sorted(
            calculated_cells, key=lambda c: c.probability, reverse=False)
        self.area_map.previous_cell = sorted_cells[0]

        return sorted_cells[0].center

    def predict_convergent_position(self, rssi_measurement: dict, reset_map: bool):
        if reset_map:
            self.area_map.reset_map()

        return self.predict_position(rssi_measurement)

    def get_map_dimensions(self, beacon_positions: dict, training_data: dict, cell_size: float):
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

        shape = np.ceil((top_corner - bottom_corner) / cell_size)

        return bottom_corner, shape


class GaussianKNNModel(GaussianProcessModel):
    """
    Implementation of gaussian model with k-nearest neighbours on the area map:
    This uses the gaussian process to produce the cell map then uses the 3 lowest cell
    centers which pass the covariance condition have high probabilities and produces
    a weighted mean of there corresponding positions.
    """

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """

        calculated_cells = self.area_map.calculate_cell_probabilities(
            rssi_measurement, self.beacons, self.prior)

        stds = np.array([cell.std for cell in calculated_cells])
        std_median = np.median(stds)
        calculated_cells = np.array(
            [cell for cell in calculated_cells if cell.std < std_median + np.std(stds)])
        sorted_cells = sorted(
            calculated_cells, key=lambda c: c.probability, reverse=False)

        self.area_map.previous_cell = sorted_cells[0]
        k = 3
        first_k = sorted_cells[:k]

        position = self.weighted_centroid_localisation(
            [cell.center for cell in first_k],
            [cell.probability for cell in first_k],
            lambda v: 1 / abs(v))

        return position


class GaussianMinMaxModel(GaussianProcessModel):
    """
    Implementation of gaussian model with k-nearest neighbours on the area map:
    This uses the gaussian process to produce the cell map then uses the 3 lowest cell
    centers which pass the covariance condition have high probabilities and then uses the bounding box method to
    """

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """

        calculated_cells = self.area_map.calculate_cell_probabilities(
            rssi_measurement, self.beacons, self.prior)

        stds = np.array([cell.std for cell in calculated_cells])
        std_median = np.median(stds)
        calculated_cells = np.array(
            [cell for cell in calculated_cells if cell.std <= std_median + np.std(stds)])
        sorted_cells = sorted(
            calculated_cells, key=lambda c: c.probability, reverse=False)

        self.area_map.previous_cell = sorted_cells[0]

        k = 3
        first_k = sorted_cells[:k]

        position = self.bounding_box([cell.center for cell in first_k], [
                                     cell.probability for cell in first_k], lambda v: 1 / abs(v))

        return position


class WKNN(BaseModel):
    """
    Implementation of a weighted k-nearest neighbours model:
    The kNN algorithm uses k = 3 and comapares the input measurement to all training data measurement, then
    uses the three closest (lowest root mean square error(RMSE)) measurements and produces
    a weighted mean of there corresponding positions.
    """

    def __init__(self, training_data_filepath: Path, filter: bool = False, k: int = 3):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        self.training_data = process_training_data(
            training_data, type=const.MeasurementProcess.QUANTILE, filter=filter)
        self.k = k

    def predict_position(self, rssi_measurement: dict, ):
        """
        Predicts the position of the target device
        """
        distances = {}
        for beacon, data in self.training_data.items():

            for line in data:
                d_hash = gh.hash_2D_coordinate(*line[1:])
                if d_hash not in distances.keys():
                    # (position,distance_scratch,beacons_used)
                    distances[d_hash] = [np.array(line[1:]), 0, 1 * 10**-6]
                if beacon in rssi_measurement.keys():
                    distances[d_hash][1] += np.square(
                        line[0] - rssi_measurement[beacon])
                    distances[d_hash][2] += 1

        for h, data in distances.items():
            distances[h][1] = np.sqrt(data[1] / distances[h][2])

        sorted_points = sorted(list(distances.values()), key=lambda p: p[1])
        first_k = np.array(sorted_points[:self.k])

        position = self.weighted_centroid_localisation(
            [point for point, _, _ in first_k],
            [distance for _, distance, _ in first_k],
            lambda v: 1 / v if v != 0 else 1 * 10 ** 10)

        return position


class KNN(BaseModel):
    """
    Implementation of a k-nearest neighbours model:
    The kNN algorithm uses k = 3 and takes the three strongest signals and produces
    a weighted mean of the corresponding beacon positions.
    """

    def __init__(self, training_data_filepath: Path, k: int = 3):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        self.k = k

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """

        rssi_measurement = {
            address: value for address, value in rssi_measurement.items()
            if address in self.beacon_positions.keys()}

        sorted_points = sorted(
            list(rssi_measurement.items()), key=lambda p: p[1], reverse=True)
        first_k = sorted_points[:self.k]

        position = self.weighted_centroid_localisation(
            [self.beacon_positions[beacon] for beacon, _ in first_k],
            [rssi for _, rssi in first_k],
            lambda v: abs(1 / v) if v != 0 else 1 * 10 ** 10)

        return position


class PropagationModel(BaseModel):
    """
    Implementation of a propagation model:
    The propagation model, uses a linear path loss model to predict distances from beacons
    and co-locates the user with them.
    """

    def __init__(self, training_data_filepath: Path, n: float, filter: bool = False):
        self.beacon_positions, training_data = fh.load_training_data(
            training_data_filepath, windows=True)
        training_data = process_training_data(
            training_data, type=const.MeasurementProcess.MEAN, filter=filter)

        # get constant values closest to 1
        beacon_constants = {}
        for beacon, data in training_data.items():
            beacon_position = self.beacon_positions[beacon]
            distances = np.linalg.norm(data[:, 1:] - beacon_position, axis=1)
            distances = distances[distances >= 1]
            closest_index = np.argmin(distances)
            beacon_constants[beacon] = (
                data[closest_index, 0], distances[closest_index])

        self.distance_functions = {}
        for beacon, constants in beacon_constants.items():
            self.distance_functions[beacon] = lambda rssi: constants[1] * \
                np.power((rssi - constants[0]) / -10 * n, 10)

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """

        distance_sum = 1 * 10**-6
        beacon_distances = {}

        rssi_measurement = {
            address: value for address, value in rssi_measurement.items()
            if address in self.beacon_positions.keys()}

        for beacon, measurement in rssi_measurement.items():
            distance = self.distance_functions[beacon](measurement)
            distance_sum += distance
            beacon_distances[beacon] = distance

        beacon_positions = np.array(
            [self.beacon_positions[beacon] for beacon in beacon_distances.keys()])

        position = self.bounding_box(beacon_positions, np.array(
            list(beacon_distances.values())), lambda v: abs(1 / v))

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

    def predict_position(self, rssi_measurement: dict):
        """
        Predicts the position of the target device
        """
        rssi_measurement = {
            address: value for address, value in rssi_measurement.items()
            if address in self.beacon_positions.keys()}

        max_beacon = max(rssi_measurement, key=rssi_measurement.get)
        return self.beacon_positions[max_beacon]
