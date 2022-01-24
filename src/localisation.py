from ast import While
import logging
from pathlib import Path
from random import sample
import numpy as np
from beacon import Beacon, create_beacons
from map import Map
from enum import Enum
from constants import Prior, MapAttribute,PROPAGATION_CONSTANT


from measurement import get_live_measurement, load_training_data
import plotting as plot

logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)




def calculate_cell_probabilities(measurements, beacons, area_map, previous_cell=None, prior=Prior.UNIFORM):
    standard_deviation = 1

    cells = area_map.get_cells
    for cell in cells:
        distance = 0.
        n = len(measurements)
        position = cell.center

        p = 1/9 if prior is Prior.LOCAL and previous_cell is not None and previous_cell.isNeighbor(
            cell) else 1/len(cells)

        if n > 0:
            beacons_used = 0
            for address, measurement in measurements.items():
                if address in beacons.keys():
                    predicted_rssi = beacons[address].predict_rssi(position)
                    distance += (measurement - predicted_rssi)**2
                    beacons_used += 1

            distance = np.sqrt(distance/beacons_used)

            p += np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))

        #print(f"cell_position: {position}  distance: {distance} p: {p}")
        cell.probability = p

        #plot_map_attribute(area_map, MapAttribute.PROB)
        #plot_map_attribute(area_map, MapAttribute.COV)


    return cells


def main():

    training_data_filepath = Path("data/intel_indoor_training.txt")
    training_data = load_training_data(training_data_filepath)

    beacons = create_beacons(training_data)

    starting_point = [-10, -10]
    ending_point = [20, 20]


    area_map = Map(starting_point, ending_point, cell_size=1)
    previous_cell = None
    previous_measurement = None

    while True:
        current_measurement = get_live_measurement(
            training_data.keys(), previous_measurement)

        rssi_measurement = {beacon: reading[0]
                            for beacon, reading in current_measurement.items()}

        print(f"rssi measurement is: {rssi_measurement}")

        cells = calculate_cell_probabilities(
            rssi_measurement, beacons, area_map, previous_cell=previous_cell, prior=Prior.UNIFORM)

        sorted_cells = sorted(cells, key=lambda c: c.probability, reverse=True)
        for i, cell in enumerate(sorted_cells[:3]):
            print(
                f"{i}. Cell: Center:{cell.center} Prob: {cell.probability} Cov: {cell.covariance}")

        previous_cell = sorted_cells[0]
        previous_measurement = current_measurement


if __name__ == "__main__":
    main()
