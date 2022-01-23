from ast import While
import logging
from pathlib import Path
import numpy as np
from beacon import Beacon
from map import Map
from enum import Enum
from constants import Prior, MapAttribute,PROPAGATION_CONSTANT


from measurement import get_live_measurement, load_training_data
from plotting import plot_beacon_map_rssi, plot_beacon_map_covariance, plot_map_attribute


logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


def create_beacon_survey_maps(training_data: dict):
    maps = {}
    for beacon, observations in training_data.items():
        maps[beacon] = Beacon(beacon,observations)

    return maps


def calculate_cell_probabilities(measurements, beacons, area_map, previous_cell=None, prior=Prior.UNIFORM):
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
            beacons_used = 0
            for address, measurement in measurements.items():
                if address in beacons.keys():
                    beacon = beacons[address]
                    predicted_cell_rssi, cell_cov = beacon.get_map.predict(
                        [position], return_cov=True)
                    

                    if cell_cov[0] < 0.1: # for sparse areas
                        predicted_cell_rssi = beacon.get_offset_rssi(position)
                    else:
                        predicted_cell_rssi = predicted_cell_rssi[0] # unpacking

                    distance += (measurement - predicted_cell_rssi[0])**2
                    beacons_used += 1

            distance = np.sqrt(distance/beacons_used)

            #print(f"cell covariance : {covariance_sum}")

            p += np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))

        #print(f"cell_position: {position}  distance: {distance} p: {p}")
        cell.probability = p
        cell.covariance = covariance_sum

        #plot_map_attribute(area_map, MapAttribute.PROB)
        #plot_map_attribute(area_map, MapAttribute.COV)

        if p > max_cell_prob[1]:
            max_cell_prob = (cell, p)

    return cells, max_cell_prob


def main():

    training_data_filepath = Path("data/intel_indoor_training.txt")
    training_data = load_training_data(training_data_filepath)

    beacons = create_beacon_survey_maps(training_data)

    starting_point = [-10, -10]
    ending_point = [20, 20]

    for beacon, map in beacons.items():
        plot_beacon_map_rssi(beacon, map, starting_point, ending_point)
        #plot_beacon_map_covariance(beacon, map, starting_point, ending_point)


    area_map = Map(starting_point, ending_point, cell_size=1)
    previous_cell = None
    previous_measurement = None

    while True:
        current_measurement = get_live_measurement(
            training_data.keys(), previous_measurement)

        rssi_measurement = {beacon: reading[0]
                            for beacon, reading in current_measurement.items()}

        print(f"rssi measurement is: {rssi_measurement}")

        cells, most_likely_cell = calculate_cell_probabilities(
            rssi_measurement, beacons, area_map, previous_cell=previous_cell, prior=Prior.UNIFORM)

        sorted_cells = sorted(cells, key=lambda c: c.probability, reverse=True)
        for i, cell in enumerate(sorted_cells[:3]):
            print(
                f"{i}. Cell: Center:{cell.center} Prob: {cell.probability} Cov: {cell.covariance}")

        previous_cell = most_likely_cell[0]
        previous_measurement = current_measurement


if __name__ == "__main__":
    main()
