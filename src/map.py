import random
import numpy as np
from constants import Prior


class Cell:

    def __init__(self, center, cell_length=1) -> None:
        self._center = center
        self._cell_length = cell_length
        self._probability = 0
        self._std = 0

    @property
    def corners(self):
        offset = self._cell_length / 2
        corner_vectors = np.array(
            [[0, offset], [offset, 0], [0, -offset], [-offset, 0]])
        return self._center + corner_vectors

    @property
    def center(self):
        return self._center

    def center_hash(self):
        tmp = self._center[1] + (self._center[0] + 1) / 2
        return self._center + tmp * tmp

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, prob):
        self._probability = prob

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        self._std = std

    def isNeighbor(self, cell):
        return np.linalg.norm(cell.center - self._center) <= np.sqrt(2 * self._cell_length)

    def __str__(self) -> str:
        return f"Cell: Center:{self._center} Prob: {self._probability} Cov: {self._std}"

    def calculate_probabilty(self, measurements, beacons, previous_cell=None, prior=Prior.UNIFORM,
                             standard_deviation=1):
        distance = 0.
        position = self._center
        log_p = np.inf
        self.std = 0

        prior_condition = (prior is Prior.LOCAL and previous_cell is not None and previous_cell.isNeighbor(
            self)) or prior is Prior.UNIFORM

        if len(measurements) > 0 and prior_condition:
            beacons_used = 0
            for address, measurement in measurements.items():
                if address in beacons.keys():
                    predicted_rssi, cell_cov = beacons[address].predict_rssi(
                        position)
                    distance += (measurement - predicted_rssi) ** 2
                    self.covariance += cell_cov[0]
                    beacons_used += 1

            distance = np.sqrt(distance / beacons_used)
            log_p = np.exp2(distance) / (2 * np.exp2(standard_deviation))
        return log_p


class Map():

    def __init__(self, bottom_corner, shape, cell_size=1) -> None:
        """
        Arguments:
        bottom_corner: n-dimensional array of the starting point ie [0,0] only supports 2/3 atm
        shape: n-dimensional array representing the total map space ie (30,30)
        cell_size: the size of each cell 
        """

        if len(bottom_corner) != len(shape):
            raise ValueError(
                f"Dimension mismatch: len(bottom_corner) ({len(bottom_corner)}) != len(shape) ({len(shape)})")

        self._cells = []
        self._bottom_corner = np.array(bottom_corner)
        self._shape = np.array(shape)

        self._previous_cell = None
        self._cell_size = cell_size
        self.cell_centers = np.empty((0, len(self._shape)))

        starting_point = self._bottom_corner + cell_size/2
        ending_point = self._bottom_corner + cell_size*self._shape

        for i in np.arange(starting_point[0], ending_point[0], step=self._cell_size):
            for j in np.arange(ending_point[1], starting_point[1], step=-self._cell_size):
                if len(self._shape) == 3:
                    for k in np.arange(ending_point[2], starting_point[2], step=-self._cell_size):
                        center = np.array([i, j])
                        self._cells.append(Cell(center, self._cell_size))
                        self.cell_centers = np.append(
                            self.cell_centers, np.array([center]), axis=0)
                else:
                    center = np.array([i, j])
                    self._cells.append(Cell(center, self._cell_size))
                    self.cell_centers = np.append(
                        self.cell_centers, np.array([center]), axis=0)

    def add_new_cells(self, new_cells):
        for cell in new_cells:
            self._cells.append(cell)

    @property
    def get_shape(self):
        return self._shape

    @property
    def get_cells(self):
        return self._cells

    @property
    def get_cell_size(self):
        return self._cell_size

    @property
    def previous_cell(self):
        return self._previous_cell

    @previous_cell.setter 
    def previous_cell(self, cell:Cell):
        self._previous_cell = cell

    def reset_map(self):
        self._previous_cell = None

    def calculate_cell_probabilities(self, measurements, beacons, prior=Prior.UNIFORM):
        standard_deviation = 2

        distance_sum = np.zeros(len(self.cell_centers))
        std_sum = np.zeros(len(self.cell_centers))
        beacons_used = {address: beacon for address,
                        beacon in beacons.items() if address in measurements.keys()}
        for address, beacon in beacons_used.items():
            rssi_predictions, std_predictions = beacon.predict_rssi(
                self.cell_centers)
            distance_sum += np.square(measurements[address] - rssi_predictions)
            std_sum += std_predictions

        distances = np.sqrt(distance_sum / len(beacons_used))
        log_p = np.exp2(distances) / (2 * np.exp2(standard_deviation))

        for i, cell in enumerate(self._cells):
            prior_condition = (prior is Prior.LOCAL and self.previous_cell is not None and self.previous_cell.isNeighbor(
                cell)) or random.randint(0,9) < 1 or prior is Prior.UNIFORM
            cell.probability = log_p[i] if prior_condition else 1*10**9
            cell.std = std_sum[i]

        return self._cells
