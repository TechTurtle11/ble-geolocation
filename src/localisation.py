import imp
import logging
from pathlib import Path
import numpy as np
import gpflow
from map import Map
from enum import Enum
import tensorflow as tf

from measurement import get_live_measurement, load_training_data


logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


class Prior(Enum):
    UNIFORM = 1
    LOCAL = 2

def create_beacon_survey_maps(training_data: dict):
    maps = {}
    for beacon, observations in training_data.items():
        Y = tf.convert_to_tensor(observations.T[0])
        X = tf.convert_to_tensor(observations.T[1:].T)
        print(Y)
        print(X)
        k = gpflow.kernels.SquaredExponential(lengthscales=[1,1])
        m = gpflow.models.GPR(data=(X,Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        print(opt_logs)
        gpflow.utilities.print_summary(m)
        maps[beacon] = m


        print(m.predict_y(Xnew=np.array([[0.5,0.5]])))

    return maps


def calculate_cell_probabilities(measurements, beacon_survey_maps, area_map,previous_cell = None, prior = Prior.UNIFORM):
    standard_deviation = 1
    max_prob = 0
    max_cell = None

    cells = area_map.get_cells
    for cell in cells:
        distance = 0.
        n = len(measurements)

        if prior is Prior.LOCAL and previous_cell.isNeighbor(cell):
            p = 1/9
        elif prior is Prior.UNIFORM:
            p = 1/len(cells)


        if n > 0:
            position = cell.center
            for beacon, map in beacon_survey_maps.items():
                cell_mean, cell_variance = map.predict_y(position)
                print(cell_mean)
                distance += (measurements[beacon] -cell_mean)**2
            distance = distance/n

            p *= np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))
        

        cell.probability = p
        

        if p > max_prob:
            max_prob = p
            max_cell = cell


    return cells, max_cell


def main():

    training_data_filepath = Path("data/test_training.txt")
    training_data = load_training_data(training_data_filepath)

    print(training_data)
    survey_maps = create_beacon_survey_maps(training_data)

    area_map = Map(initial_dimensions=(20,20))


    while True:
        current_measurement = get_live_measurement(training_data.keys())
        print(f"Current measurement is: {current_measurement}")

        cells, most_likely_cell = calculate_cell_probabilities(
            current_measurement, survey_maps, area_map)


        print(f"Most Likely Cell: Center:{most_likely_cell.center} Prob: {most_likely_cell.probability}")


if __name__ == "__main__":
    main()