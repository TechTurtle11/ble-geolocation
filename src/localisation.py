import logging
from pathlib import Path
import numpy as np
from map import Map
from constants import Prior
import general_helper as gh
import file_helper as fh
from beacon import create_beacons
from measurement import get_live_measurement, process_training_data

logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)




def calculate_cell_probabilities(measurements, beacons, area_map, previous_cell=None, prior=Prior.UNIFORM):
    standard_deviation = 2

    cells = area_map.get_cells
    for cell in cells:
        cell.probability = cell.calculate_probabilty(measurements,beacons, previous_cell,prior, standard_deviation)
        logging.debug(f"{cell.center} {cell.probability}")

    return cells



def run_iteration(beacons, area_map, previous_measurement = None, previous_cell= None, prior=Prior.UNIFORM):
    current_measurement = get_live_measurement(previous_measurement)

    rssi_measurement = {beacon: reading[0]
                        for beacon, reading in current_measurement.items()}

    print(f"rssi measurement is: {rssi_measurement}")
    logging.debug(f"rssi measurement is: {rssi_measurement}")

    cells = calculate_cell_probabilities(
        rssi_measurement, beacons, area_map, previous_cell, prior)

    sorted_cells = sorted(cells, key=lambda c: c.probability, reverse=False)
    for i, cell in enumerate(sorted_cells[:3]):
        print(f"{i}. {cell}")
    logging.debug(f"Most likely cell: {sorted_cells[0]}")
    return sorted_cells[0], current_measurement


def predict_positions(training_data_filepath,iterations,prior):


    predicted_positions = {}

    while True:
        x = input("Enter current x coordinate: ")
        y = input("Enter current y coordinate: ")
        position = np.array([float(x),float(y)])

        cells = run_localisation_iterations(training_data_filepath, iterations,prior)           


        predicted_positions[gh.hash_2D_coordinate(*position)] = [position,cells]

        loop_continue = input("Type stop if you have finished collecting training data: ")
        if "stop" in loop_continue.lower():
            break

    return predicted_positions

def predict_and_write_positions(training_data_filepath,write_filepath,prior):
    pred = predict_positions(training_data_filepath,10,prior)
    fh.write_position_prediction_to_file(pred,write_filepath)


def run_localisation_iterations(training_data_filepath, iterations, prior):
    training_data_filepath = Path(training_data_filepath)
    training_data = fh.load_training_data(training_data_filepath,windows=True)
    training_data = process_training_data(training_data)

    beacons = create_beacons(training_data)

    starting_point = [-10, -10]
    ending_point = [20, 20]


    area_map = Map(starting_point, ending_point, cell_size=1)

    selected_cells = []

    previous_cell = None
    previous_measurement = None
    for _ in  range(iterations):
        previous_cell,previous_measurement = run_iteration(beacons,area_map,previous_measurement,previous_cell, prior)
        selected_cells.append(previous_cell)

    return selected_cells


def main():

    training_data_filepath = Path("data/intel_indoor_training_2.txt")
    position_prediction_filepath = Path("data/predictions/test.txt")
    predict_and_write_positions(training_data_filepath, position_prediction_filepath, Prior.UNIFORM)

if __name__ == "__main__":
    main()
