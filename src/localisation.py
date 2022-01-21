import imp
import logging
from pathlib import Path
import numpy as np
import gpflow
from map import Map
from enum import Enum
import tensorflow as tf
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

        kernel = ConstantKernel() + RBF()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,normalize_y=True).fit(X, Y)

        print(gpr.predict(np.array([[.5,0.5]])))


        maps[beacon] = gpr


    return maps


def calculate_cell_probabilities(measurements, beacon_survey_maps, area_map,previous_cell = None, prior = Prior.UNIFORM):
    standard_deviation = 1
    max_prob = 0
    max_cell = None

    cells = area_map.get_cells
    for cell in cells:
        distance = 0.
        n = len(measurements)

        if prior is Prior.LOCAL and not previous_cell is None and previous_cell.isNeighbor(cell):
            p= 1/9
        elif prior is Prior.LOCAL and not previous_cell is None:
            p = 1/len(cells)
        else:
            p = 1/len(cells)


        if n > 0:
            position = cell.center
            for beacon, map in beacon_survey_maps.items():
                cell_mean = map.predict([position])
                distance += (measurements[beacon] -cell_mean)**2
            distance = np.sqrt(distance/n)
            

            p += np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))
        print(f"cell_position: {position}  distance: {distance} p: {p}")
        cell.probability = p
        

        if p > max_prob:
            max_prob = p
            max_cell = cell


    return cells, max_cell


def main():

    training_data_filepath = Path("data/cl_indoor_training.txt")
    training_data = load_training_data(training_data_filepath)

    survey_maps = create_beacon_survey_maps(training_data)

    area_map = Map(initial_dimensions=[10,10])
    previous_cell = None


    while True:
        current_measurement = get_live_measurement(training_data.keys())
        print(f"Current measurement is: {current_measurement}")

        cells, most_likely_cell = calculate_cell_probabilities(
            current_measurement, survey_maps, area_map,previous_cell=previous_cell, prior=Prior.LOCAL)


        print(f"Most Likely Cell: Center:{most_likely_cell.center} Prob: {most_likely_cell.probability}")

        previous_cell = most_likely_cell

if __name__ == "__main__":
    main()