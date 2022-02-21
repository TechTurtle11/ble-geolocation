import numpy as np

from constants import Prior


class Cell:

    def __init__(self, center, cell_length=1) -> None:
        self._center = center
        self._cell_length = cell_length
        self._probability = 0
        self._covariance = 0

    @property
    def corners(self):
        offset = self._cell_length / 2
        corner_vectors = np.array([[0, offset], [offset, 0], [0, -offset], [-offset, 0]])
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
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov

    def isNeighbor(self, cell):
        return np.linalg.norm(cell.center - self._center) <= np.sqrt(2 * self._cell_length)

    def __str__(self) -> str:
        return f"Cell: Center:{self._center} Prob: {self._probability} Cov: {self._covariance}"

    def calculate_probabilty(self, measurements, beacons, previous_cell=None, prior=Prior.UNIFORM,
                             standard_deviation=1):
        distance = 0.
        position = self._center
        log_p = np.inf
        self.covariance = 0

        prior_condition = (prior is Prior.LOCAL and previous_cell is not None and previous_cell.isNeighbor(
            self)) or prior is Prior.UNIFORM

        if len(measurements) > 0 and prior_condition:
            beacons_used = 0
            for address, measurement in measurements.items():
                if address in beacons.keys():
                    predicted_rssi,cell_cov = beacons[address].predict_rssi(position)
                    distance += (measurement - predicted_rssi) ** 2
                    self.covariance += cell_cov[0]
                    beacons_used += 1

            distance = np.sqrt(distance / beacons_used)
            log_p = np.exp2(distance) / (2 * np.exp2(standard_deviation))
        return log_p


class Map():

    def __init__(self, starting_point=None, ending_point=None, cell_size=1) -> None:
        self._cells = []
        self._dimensions = (starting_point, ending_point)
        self._cell_size = cell_size
        self.cell_centers = np.empty((0,2))

        if starting_point is None:
            starting_point = (0.5, 0.5)



        if ending_point is not None:
            for i in np.arange(start=min(starting_point), stop=max(ending_point), step=self._cell_size):
                for j in np.arange(start=max(ending_point), stop=min(starting_point), step=-self._cell_size):
                    center = np.array([i, j])
                    self._cells.append(Cell(center, self._cell_size))
                    self.cell_centers = np.append(self.cell_centers,np.array([center]),axis=0)

    def add_new_cells(self, new_cells):
        for cell in new_cells:
            self._cells.append(cell)

    @property
    def get_dimensions(self):
        return self._dimensions

    @property
    def get_cells(self):
        return self._cells

    @property
    def get_cell_size(self):
        return self._cell_size

    def calculate_cell_probabilities(self, measurements, beacons,previous_cell=None, prior=Prior.UNIFORM):
        standard_deviation = 2

        distance_sum = np.zeros(len(self.cell_centers))
        std_sum = np.zeros(len(self.cell_centers))
        beacons_used = {address:beacon for address,beacon in beacons.items() if address in measurements.keys()}
        for address,beacon in beacons_used.items():
            rssi_predictions,covariance_predictions = beacon.predict_rssi(self.cell_centers)
            distance_sum += np.square(measurements[address] - rssi_predictions)
            std_sum += covariance_predictions

    
        distances = np.sqrt(distance_sum / len(beacons_used))
        log_p = np.exp2(distances) / (2 * np.exp2(standard_deviation))
            
        for i,cell in enumerate(self._cells):
            prior_condition = (prior is Prior.LOCAL and previous_cell is not None and previous_cell.isNeighbor(self)) or prior is Prior.UNIFORM
            cell.probability = log_p[i] if prior_condition else np.inf
            cell.covariance = std_sum[i]

        return self._cells
