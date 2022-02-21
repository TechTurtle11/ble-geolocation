import logging
from pathlib import Path
import cProfile

import numpy as np

import file_helper as fh
import general_helper as gh
from beacon import create_beacons
from constants import MapAttribute, Prior
from map import Map
from measurement import get_live_measurement, process_training_data
from models import KNN, WKNN, GaussianKNNModel, GaussianProcessModel, PropagationModel
from plotting import plot_map_attribute, produce_average_localisation_distance_plot, produce_localisation_distance_plot

logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


def calculate_cell_probabilities(measurements, beacons, area_map, previous_cell=None, prior=Prior.UNIFORM):
    standard_deviation = 2

    cells = area_map.get_cells
    for cell in cells:
        cell.probability = cell.calculate_probabilty(
            measurements, beacons, previous_cell, prior, standard_deviation)
        logging.debug(f"{cell.center} {cell.probability}")

    return cells


def run_iteration(beacons, area_map, previous_measurement=None, previous_cell=None, prior=Prior.UNIFORM):
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

    plot_map_attribute(area_map,MapAttribute.PROB)
    return sorted_cells[0], current_measurement


def run_localisation_on_file(evaluation_data_filepath,model):


    evaluation_data = fh.load_evaluation_data(evaluation_data_filepath)

    position_predictions = []
    for position,measurement in evaluation_data:
            predicted_position = model.predict_position(measurement)
            position_predictions.append((position,predicted_position))

    return position_predictions


def run_localisation_comparison(training_data_filepath, evaluation_data_filepath):


    models = {}
    models["Gaussian"] = GaussianProcessModel(training_data_filepath,prior=Prior.UNIFORM,starting_point=[-3,-3],ending_point=[10,17],cell_size=1)
    models["KNN"] = KNN(training_data_filepath,)
    models["WKNN"] = WKNN(training_data_filepath,)
    models["GaussianKNN"] = GaussianKNNModel(training_data_filepath,prior=Prior.UNIFORM,starting_point=[-3,-3],ending_point=[10,17],cell_size=1)
    models["Propagation"] = PropagationModel(training_data_filepath,2)

    print("Created models")

    predictions = {name: run_localisation_on_file(evaluation_data_filepath,model) for name,model in models.items()}


    #produce_localisation_distance_plot(predictions)
    produce_average_localisation_distance_plot(predictions)



def predict_positions(training_data_filepath, iterations, prior):
    predicted_positions = {}

    training_data_filepath = Path(training_data_filepath)
    beacon_positions, training_data = fh.load_training_data(training_data_filepath, windows=True)
    training_data = process_training_data(training_data)

    while True:
        x = input("Enter current x coordinate: ")
        y = input("Enter current y coordinate: ")
        position = np.array([float(x), float(y)])

        cells = run_localisation_iterations(
            training_data,beacon_positions, iterations, prior)

        predicted_positions[gh.hash_2D_coordinate(
            *position)] = [position, cells]

        loop_continue = input(
            "Type stop if you have finished collecting training data: ")
        if "stop" in loop_continue.lower():
            break

    return beacon_positions, predicted_positions


def predict_and_write_positions(training_data_filepath, write_filepath, prior):
    iterations = 10
    beacon_positions, pred = predict_positions(training_data_filepath, iterations, prior)
    fh.write_position_prediction_to_file(pred,beacon_positions, write_filepath)


def run_localisation_iterations(training_data, beacon_locations, iterations, prior):


    beacons = create_beacons(beacon_locations, training_data)

    starting_point = [-3, -3]
    ending_point = [10, 17]

    area_map = Map(starting_point, ending_point, cell_size=1)

    selected_cells = []

    previous_cell = None
    previous_measurement = None
    for _ in range(iterations):
        previous_cell, previous_measurement = run_iteration(
            beacons, area_map, previous_measurement, previous_cell, prior)
        selected_cells.append(previous_cell)

    return selected_cells


def main():
    training_data_filepath = Path("data/intel_indoor_training_3_edited.txt")
    position_prediction_filepath = Path("data/predictions/test1.txt")
    evaluation_data_filepath = Path("data/evaluation_intel.txt")   

    run_localisation_comparison(training_data_filepath,evaluation_data_filepath)

    #predict_and_write_positions(training_data_filepath, position_prediction_filepath, Prior.UNIFORM)


if __name__ == "__main__":
    #print(cProfile.run("main ()"))
    main()
