import argparse
import logging
from pathlib import Path

import utils.file_helper as fh
from processing.filtering import KalmanFilter
import utils.general_helper as gh
from utils.constants import Prior
from measurement import get_live_measurement, process_evaluation_data
import paradigms.models as models


logging.basicConfig(filename='logs/localisation.log', level=logging.ERROR)


def run_localisation_on_file(evaluation_data_filepath, model, filtering=True, filter=KalmanFilter):

    evaluation_data = fh.load_evaluation_data(evaluation_data_filepath)
    evaluation_data = process_evaluation_data(evaluation_data)

    position_filter_map = {}

    position_predictions = []
    for position, measurement in evaluation_data:

        h = gh.hash_2D_coordinate(*position)
        if h in position_filter_map.keys():
            filter_map = position_filter_map[h]
        else:
            filter_map = {}

        for beacon, rssi_value in measurement.items():
            if beacon not in filter_map.keys():
                filter_map[beacon] = filter()

            if filtering:
                measurement[beacon] = filter_map[beacon].predict_and_update(rssi_value)
        position_filter_map[h] = filter_map

        predicted_position = model.predict_position(measurement)
        position_predictions.append((position, predicted_position))

    return position_predictions


def run_convergence_localisation_on_file(
        evaluation_data_filepath, model, filtering=True, filter=KalmanFilter):
    evaluation_data = fh.load_evaluation_data(evaluation_data_filepath)
    evaluation_data = process_evaluation_data(evaluation_data)

    position_filter_map = {}
    reset_map = False
    position_predictions = []
    for position, measurement in evaluation_data:

        h = gh.hash_2D_coordinate(*position)
        if h in position_filter_map.keys():
            filter_map = position_filter_map[h]

        else:
            filter_map = {}
            reset_map = True

        for beacon, rssi_value in measurement.items():
            if beacon not in filter_map.keys():
                filter_map[beacon] = filter()
            else:
                if filtering:
                    measurement[beacon] = filter_map[beacon].predict_and_update(rssi_value)
        position_filter_map[h] = filter_map

        predicted_position = model.predict_convergent_position(measurement, reset_map)
        reset_map = False
        position_predictions.append((position, predicted_position))

    return position_predictions


def live_localisation(training_filepath: Path):
    """Used for live localisation: Spits out position predictions continuously

    Args:
        training_filepath (Path): The training data for the model.
    """

    # standard gaussian is used for the model without a prior
    model = models.GaussianProcessModel(
        training_filepath,
        prior=Prior.UNIFORM,
        cell_size=1,
        filter=True)

    measurement = None
    while True:
        measurement = get_live_measurement(measurement)
        stripped_measurement = {beacon: reading[0]
                                for beacon, reading in stripped_measurement.items()}

        position_prediction = model.predict_position(stripped_measurement)
        print(f"Position Prediction: {position_prediction}")
        logging.debug(position_prediction)


def adhoc_localisation(training_filepath):
    """NOT DONE : load training data, translate training data based on predicted position, add to map"""
    return None


def main():
    print("Welcome to the localisation file")

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Plotting Wanted")
    parser.add_argument(
        "file", help="The file with the training data in it.")
    args = parser.parse_args()

    modes = ["live", "adhoc",]
    if args.mode not in modes:
        print("Mode should be in " + ",".join(modes))
    else:
        training_data = Path(args.parse)

        if args.mode == "live":
            live_localisation(training_data)
        if args.mode == "adhoc":
            adhoc_localisation(training_data)


if __name__ == "__main__":
    main()
