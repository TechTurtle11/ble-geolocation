import logging
from pathlib import Path
import numpy as np
import gpflow
from map import Map

from measurement import get_live_measurement, load_training_data


logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


def create_beacon_survey_maps(training_data: dict):
    maps = {}
    for beacon, observations in training_data.items():

        k = gpflow.kernels.SquaredExponential()
        m = gpflow.models.GPR(data=observations, kernel=k, mean_function=None)
        m.kernel.lengthscales.assign(np.array([0, 0]))
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        maps[beacon] = m

    return maps


def calculate_cell_probabilities(measurements, beacon_survey_maps, area_map):
    standard_deviation = 1
    max_prob = 0
    max_cell = None

    cells = area_map.get_cells()
    for cell in cells:
        distance = 0
        n = len(measurements)
        position = cell.center
        for beacon, measurement in measurements.items():
            cell_mean, cell_variance = beacon_survey_maps[beacon].predict_f(position)

            distance += (measurement -cell_mean)**2
        distance = distance/n

        p = np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))
        cell.probability = p

        if p > max_prob:
            max_prob = p
            max_cell = cell


    return cells, max_cell


def main():

    training_data_filepath = Path("data/test_training.txt")
    training_data = load_training_data(training_data_filepath)

    survey_maps = create_beacon_survey_maps(training_data)

    area_map = Map(initial_dimensions=(20,20))


    while True:
        current_measurement = get_live_measurement(training_data.keys())

        cells, most_likely_cell = calculate_cell_probabilities(
            current_measurement, survey_maps, area_map)


        print(f"Most Likely Cell: Center:{most_likely_cell.center} Prob: {most_likely_cell.probability}")