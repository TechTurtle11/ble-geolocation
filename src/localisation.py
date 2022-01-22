import logging
from pathlib import Path
import numpy as np
from map import Map
from enum import Enum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


from measurement import get_live_measurement, load_training_data


logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


class Prior(Enum):
    UNIFORM = 1
    LOCAL = 2


def create_beacon_survey_maps(training_data: dict):
    maps = {}
    for beacon, observations in training_data.items():
        Y = observations.T[0]
        X = observations.T[1:].T

        kernel = RBF()
        gpr = GaussianProcessRegressor(
            kernel=kernel, random_state=0, normalize_y=True).fit(X, Y)

        print(f"{beacon} {gpr.predict(np.array([[0,0]]))}")

        maps[beacon] = gpr

    return maps


def calculate_cell_probabilities(measurements, beacon_survey_maps, area_map, previous_cell=None, prior=Prior.UNIFORM):
    standard_deviation = 1
    max_cell_prob = (None, 0)  # cell, probability

    cells = area_map.get_cells
    for cell in cells:
        distance = 0.
        n = len(measurements)
        position = cell.center

        p = 1/9 if prior is Prior.LOCAL and previous_cell is not None and previous_cell.isNeighbor(
            cell) else 1/len(cells)

        covariance_sum = 0
        covariance_threshold = 400
        if n > 0:
            for beacon, map in beacon_survey_maps.items():
                cell_mean, cell_cov = map.predict([position], return_cov=True)
                distance += (measurements[beacon] - cell_mean[0])**2
                covariance_sum += cell_cov[0]
            distance = np.sqrt(distance/n)
            #print(f"cell covariance : {covariance_sum}")

            p += np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))

        #print(f"cell_position: {position}  distance: {distance} p: {p}")
        cell.probability = p
        cell.covariance = covariance_sum

        if p > max_cell_prob[1]:
            max_cell_prob = (cell, p)

    return cells, max_cell_prob


def main():

    training_data_filepath = Path("data/intel_indoor_training.txt")
    training_data = load_training_data(training_data_filepath)

    survey_maps = create_beacon_survey_maps(training_data)

    area_map = Map(starting_point=[-10, -10], ending_point=[20, 20],cell_size=1)
    previous_cell = None
    previous_measurement = None

    while True:
        current_measurement = get_live_measurement(
            training_data.keys(), previous_measurement)

        rssi_measurement = {beacon:reading[0] for beacon, reading in current_measurement.items()}

        print(f"Current measurement is: {current_measurement}")

        cells, most_likely_cell = calculate_cell_probabilities(
            rssi_measurement, survey_maps, area_map, previous_cell=previous_cell, prior=Prior.UNIFORM)

        sorted_cells = sorted(cells, key=lambda c: c.probability, reverse=True)
        for cell in sorted_cells[:3]:
            print(
                f"Cell: Center:{cell.center} Prob: {cell.probability} Cov: {cell.covariance}")

        #print(f"Most Likely Cell: Center:{most_likely_cell[0].center} Prob: {most_likely_cell[0].probability}")

        previous_cell = most_likely_cell[0]
        previous_measurement = current_measurement


if __name__ == "__main__":
    main()
