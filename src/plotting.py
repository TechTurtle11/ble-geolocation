from __future__ import annotations
import argparse

from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import Utils.file_helper as fh
from Processing.filtering import BasicFilter, KalmanFilter, MovingMeanFilter, MovingMedianFilter
import measurement as measure
from Models.beacon import create_beacons
from Utils.constants import MapAttribute
from Models.map import Map


def plot_beacon_map_rssi(beacon_name, beacon, starting_point, ending_point):
    square_start = min(starting_point)
    square_end = max(ending_point)

    x_samples = np.arange(square_start, square_end, 1)
    y_samples = np.arange(square_start, square_end, 1)[::-1]

    predictions = np.array([np.array([beacon.predict_rssi(
        [np.array([x, y])])[0] for x in x_samples]) for y in y_samples]).reshape((square_end, square_end))

    predictions = np.rint(predictions).astype(int)

    fig, ax = plot_heatmap(x_samples, y_samples, predictions,
                           f"RSSI Map for beacon: {beacon_name} at {beacon.position}", "Coordinate (m)",
                           "Coordinate (m)", annotations=True, colorbar=True)


def plot_beacon_map_covariance(beacon_name, beacon_map, starting_point=[-10, -10], ending_point=[20, 20]):
    square_start = min(starting_point)
    square_end = max(ending_point)

    x_samples = np.arange(square_start, square_end, 1)
    y_samples = np.arange(square_start, square_end, 1)[::-1]
    predictions = [[beacon_map.predict([np.array([x, y])], return_cov=True)[
        1][0][0] for x in x_samples] for y in y_samples]

    plot_heatmap(x_samples, y_samples, np.array(predictions),
                 f"Covariance Map for beacon: {beacon_name}", "Coordinate (m)", "Coordinate (m)", colorbar=True)


def plot_training_data(training_data):
    x_samples = []
    y_samples = []
    rssi_values = []
    for beacon_data in training_data.values():
        for point, value in beacon_data:
            x_samples.append(point[0])
            y_samples.append(point)
            rssi_values.append(value)


def plot_heatmap(x_samples, y_samples, predictions, title, x_label, y_label, annotations=False, colorbar=False):
    fig, ax = plt.subplots()
    im = ax.imshow(predictions)

    # Setting the labels
    ax.set_xticks(np.arange(len(y_samples)))
    ax.set_yticks(np.arange(len(x_samples)))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # labeling respective list entries
    ax.set_xticklabels(x_samples)
    ax.set_yticklabels(y_samples)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    if annotations:
        # Creating text annotations by using for loop
        for i in range(len(x_samples)):
            for j in range(len(y_samples)):
                text = ax.text(j, i, predictions[i, j],
                               ha="center", va="center", color="w")
    if colorbar:
        fig.colorbar(im)

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_map_attribute(map: Map, attribute: MapAttribute):
    cells = map.get_cells
    start, end = map.get_dimensions
    cell_size = map.get_cell_size

    square_start = min(start)
    square_end = max(end)

    x_samples = np.arange(square_start, square_end, cell_size)
    y_samples = np.arange(square_start, square_end, cell_size)[::-1]

    if attribute is MapAttribute.PROB:
        title = f"Probability Heatmap "
        attribute_values = [cell.probability for cell in cells]
    elif attribute is MapAttribute.COV:
        title = f"Covariance Heatmap "
        attribute_values = [cell.covariance for cell in cells]
    else:
        raise ValueError(f"{attribute} isn't supported")

    attribute_values = np.array(attribute_values).reshape(
        (len(x_samples), len(y_samples))).T

    fig, ax = plot_heatmap(x_samples, y_samples, attribute_values,
                           title, "Coordinate (m)", "Coordinate (m)", colorbar=True)


def plot_rssi_distance(beacon, beacon_location, predict=False):
    fig, ax = plt.subplots()
    data = beacon.training_data

    rssi_values = data.T[0]
    distances = np.linalg.norm(data.T[1:].T - beacon_location, axis=1)

    predicted_rssi_values = beacon.predict_rssi(data.T[1:].T)

    plt.xlabel("Distance(m)")
    plt.ylabel("RSSI Value(dBm)")
    plt.title(f"Plot showing rssi values with distance for {beacon}")

    plt.scatter(distances, rssi_values)
    if predict:
        plt.scatter(distances, predicted_rssi_values)


def plot_rssi_readings_over_time(data_set, title="unknown"):
    plot_comparison(data_set, "Time (reading)", "RSSI Values(dBm)", title)


def plot_comparison(data_sets, x_axis, y_axis, title):
    fig, ax = plt.subplots()

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    for name, readings in data_sets.items():
        plt.plot(readings, label=name)

    plt.legend(loc="lower right")


def plot_position_prediction(position, predicted_cells, beacons):
    predicted_positions = np.array([cell.center for cell in predicted_cells])
    beacon_positions = np.array(list(beacons.values()))

    fig, ax = plt.subplots()

    ax.set_xlabel("Coordinate (m)")
    ax.set_ylabel("Coordinate (m)")
    ax.set_title(f"Plot showing position prediction for {position}")

    ax.scatter(beacon_positions[:, 0], beacon_positions[:, 1], label="Beacons")
    ax.scatter(position[0], position[1], label="Position to Predict")
    ax.scatter(
        predicted_positions[:, 0], predicted_positions[:, 1], label="Position Predictions")
    ax.legend()
    plt.show()


def plot_filtered_rssi_comparison(measurement, title,filters, round=False):

    filtered_lists = {k: f.filter_list(measurement)
                      for k, f in filters.items()}

    if round:
        filtered_lists = {k: np.round(ls) for k, ls in filtered_lists.items()}

    data_set = {"non-filtered": measurement}
    data_set.update(filtered_lists)
    plot_rssi_readings_over_time(data_set, title)


def produce_position_prediction_plots(filepath):
    beacon_positions, predictions = fh.read_position_prediction_from_file(
        filepath)
    for position, cells in predictions.values():
        plot_position_prediction(position, cells, beacon_positions)


def produce_measurement_plots(measurement_filepath, round=False):
    measurement = fh.read_measurement_from_file(measurement_filepath)
    mean_measurement = np.array([np.mean(window) for window in measurement])
    median_measurement = np.array([np.median(window) for window in measurement])

    flattened_readings = np.array(list(chain.from_iterable(measurement)))

    print(f"flatstd {np.std(median_measurement)}")
    print(f"mean{np.mean(median_measurement)}")
    print(f"median {np.median(median_measurement)}")


    plot_rssi_readings_over_time(
        {"mean": mean_measurement, "median": median_measurement}, "Raw RSSI values over time")
    #plt.hist(flattened_readings, bins=40)
    plot_rssi_readings_over_time(
        {"raw rssi measurements": flattened_readings}, "Mean vs Median")

    filters = {"basic": BasicFilter(), "Kalman": KalmanFilter(
    ), "Mean": MovingMeanFilter(), "Median": MovingMedianFilter()}

    plot_filtered_rssi_comparison(
        mean_measurement[100:700], "Filter Comparison",{"Median": MovingMedianFilter(), "Kalman": KalmanFilter()}, round)

    plot_filtered_rssi_comparison(measurement[0], "Window comparison",filters, round)
    plot_filtered_rssi_comparison(
        mean_measurement[100:700], "Mean Filter comparison",filters, round)
    plot_filtered_rssi_comparison(
        flattened_readings, "Raw filter comparison",filters, round)
    plt.show()


def produce_beacon_map_plots(training_data_filepath, starting_point, ending_point):
    beacon_positions, training_data = fh.load_training_data(
        training_data_filepath, windows=True)
    training_data = measure.process_training_data(training_data)
    beacons = create_beacons(beacon_positions, training_data)

    for address, beacon in beacons.items():
        plot_rssi_distance(beacon, beacon.position)
        plot_beacon_map_rssi(address, beacon, starting_point,
                             ending_point)
        plot_beacon_map_covariance(
            address, beacon.get_map, starting_point, ending_point)
        plt.show()


def produce_rotation_plot():
    measurement_filepaths = {angle: Path(
        f"data/experiment/test_rotation_{angle}_measurement.csv") for angle in [0, 90, 180, 270, ]}
    measurements = {angle: fh.read_measurement_from_file(
        filepath) for angle, filepath in measurement_filepaths.items()}

    filters = {angle: MovingMeanFilter() for angle, _ in measurements.items()}

    measurements = {angle: filters[angle].filter_list(np.array(list(chain.from_iterable(
        measurement)))) for angle, measurement in measurements.items()}
    plot_rssi_readings_over_time(measurements, "RSSI by angle of rotation")


def produce_localisation_distance_plot(algorithm_predictions):
    distances = {}
    for algorithm, predictions in algorithm_predictions.items():
        distances[algorithm] = [np.linalg.norm(
            actual - prediction) for actual, prediction in predictions]

    fig, ax = plt.subplots()

    labels = {str(actual)
              for actual, _ in list(algorithm_predictions.values())[0]}
    filtered_distances = {}
    for algorithm, distance_values in distances.items():

        for label in labels:
            filtered_label_distances = [distance for i, distance in enumerate(
                distance_values) if str(algorithm_predictions[algorithm][i][0]) == label]
            if not algorithm in filtered_distances:
                filtered_distances[algorithm] = []
            filtered_distances[algorithm].append(
                sum(filtered_label_distances) / len(filtered_label_distances))

    filtered_distances = dict(
        sorted(filtered_distances.items(), key=lambda d: d[1]))

    ax.set_xlabel("Positions ")
    ax.set_ylabel("Distance From actual point")
    ax.set_title(
        f"Barchart to demonstrate algorithm prediction accuracy for certain points")

    br = np.arange(len(labels))
    bar_width = 0.175
    plt.xticks(ticks=br+bar_width, labels=list(labels))

    for i, algorithm in enumerate(filtered_distances.keys()):
        ax.bar(br+bar_width*i,
               filtered_distances[algorithm], label=algorithm, width=bar_width)

    ax.legend()
    plt.show()


def produce_average_localisation_distance_plot(algorithm_predictions):
    distances = {}
    std = {}
    medians = {}
    for algorithm, predictions in algorithm_predictions.items():
        dist = [np.linalg.norm(actual - prediction)
                for actual, prediction in predictions]
        distances[algorithm] = np.mean(dist)
        medians[algorithm] = np.median(dist)
        std[algorithm] = np.std(dist)

    distances = dict(sorted(distances.items(), key=lambda d: d[1]))

    fig, ax = plt.subplots()

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Average Error From actual point (m)")
    ax.set_title(f"Average Error for localisation algorithms")

    bar_width = 0.3

    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y", zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':',
            linewidth=0.5, axis="y", zorder=0)
    ax.minorticks_on()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    bars = []
    for i, algorithm in enumerate(distances.keys()):
        bar = ax.bar(bar_width*i, distances[algorithm], yerr=std[algorithm],
                     label=algorithm, width=bar_width, zorder=3)
        for i, rect in enumerate(bar):
            height = std[algorithm] + rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                     f'{distances[algorithm]:.2f}', ha='center', va='bottom', zorder=4)

    ax.legend()
    plt.show()


def parameter_plot(predictions,parameter):
    y_label = "Mean Average Error (m)"

    if parameter == "cell_size":
        x_label = "Cell Size (m)"
        title = f"How Cell Size Affects Gaussian Model Performance"
    elif parameter == "k":
        x_label = "K"
        title = f"How K affects KNN Model Performance"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


    x = list(predictions.keys())
    y = [p[0] for p in predictions.values()]
    ci = [p[1] for p in predictions.values()]
    ax.errorbar(x,y,yerr=ci)
    plt.show()

    

def comparison_plot(filtered,unfiltered,metric):
    x_label = "Algorithm"

    y_label = "Mean Average Error (m)"
    title = f"{y_label} for Localisation Algorithms"


    if metric == "filter":
        applied = "Filtered"
        unapplied = "Unfiltered"
        title = f"Filter Comparison for Localisation Algorithms"
    elif metric == "prior":
        applied = "Local"
        unapplied = "Uniform"
        title = f"Prior Comparison for Localisation Algorithms"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label,fontsize=12)
    ax.set_ylabel(y_label,fontsize=12)
    ax.set_title(title,fontsize = 14)

    bar_width = 0.3

    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y", zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':',
            linewidth=0.5, axis="y", zorder=0)
    ax.minorticks_on()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True, # labels along the bottom edge are off
        )  

    xtick_locations = (2*bar_width+0.1) * np.arange(len(filtered.keys())) + bar_width/2
    ax.set_xticks(xtick_locations)
    ax.set_xticklabels(list(filtered.keys()),rotation="30",ha="right")

    for i, algorithm in enumerate(filtered.keys()):
        if i == 0:
            fbar = ax.bar(bar_width*(i)*2+0.1*i, filtered[algorithm][0],
                            yerr=filtered[algorithm][1], label=applied, width=bar_width, zorder=3,color="blue")
            
            ufbar = ax.bar(bar_width*(i*2+1)+0.1*i, unfiltered[algorithm][0],
                            yerr=unfiltered[algorithm][1], label=unapplied, width=bar_width, zorder=3,color="green")
        else:
            fbar = ax.bar(bar_width*(i)*2+0.1*i, filtered[algorithm][0],
                            yerr=filtered[algorithm][1], width=bar_width, zorder=3,color="blue")
            
            ufbar = ax.bar(bar_width*(i*2+1)+0.1*i, unfiltered[algorithm][0],
                            yerr=unfiltered[algorithm][1], width=bar_width, zorder=3,color="green")


        for i, rect in enumerate(fbar):
            height = filtered[algorithm][1] + rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                        f'{filtered[algorithm][0]:.2f}', ha='center', va='bottom', zorder=4)
        for i, rect in enumerate(ufbar):
            height = unfiltered[algorithm][1] + rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                        f'{unfiltered[algorithm][0]:.2f}', ha='center', va='bottom', zorder=4)

    ax.legend(loc="upper left")
    plt.show()

def plot_evaluation_metric(predictions, metric):

    x_label = "Algorithm"
    if metric == "mae":
        y_label = "Mean Average Error (m)"
        title = f"{y_label} for Localisation Algorithms"
    elif metric == "rmse":
        y_label = "Root Mean Square Error (m)"
        title = f"{y_label} for Localisation Algorithms"
    elif metric == "mae,cell_size":
        y_label = "Mean Average Error (m)"
        x_label = "Cell Size (m)"
        title = f"How Cell Size Affects Gaussian Model Performance"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label,fontsize=12)
    ax.set_ylabel(y_label,fontsize=12)
    ax.set_title(title,fontsize = 14)

    bar_width = 0.3

    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y", zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':',
            linewidth=0.5, axis="y", zorder=0)
    ax.minorticks_on()

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True, # labels along the bottom edge are off
        )  

    xtick_locations = bar_width * np.arange(len(predictions.keys()))
    ax.set_xticks(xtick_locations)
    ax.set_xticklabels(list(predictions.keys()),rotation="30",ha="right")

    for i, algorithm in enumerate(predictions.keys()):
        if len(list(predictions.values())[0]) > 1:
            bar = ax.bar(bar_width*i, predictions[algorithm][0],
                         yerr=predictions[algorithm][1], label=algorithm, width=bar_width, zorder=3)
        else:
            bar = ax.bar(
                bar_width*i, predictions[algorithm], label=algorithm, width=bar_width, zorder=3)

        for i, rect in enumerate(bar):
            if len(list(predictions.values())[0]) > 1:
                height = predictions[algorithm][1] + rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                         f'{predictions[algorithm][0]:.2f}', ha='center', va='bottom', zorder=4)

            else:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                         f'{predictions[algorithm]:.2f}', ha='center', va='bottom', zorder=4)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Plotting Wanted")
    parser.add_argument(
        "file", help="The file with the training data in it.")
    args = parser.parse_args()

    modes = ["rotate", "measure", "beacon","position"]
    if args.mode not in modes:
        print("Mode should be in " + ",".join(modes))
    else:
        if args.mode == "rotate":
            produce_rotation_plot()
        if args.mode == "measure":
            filepath = Path(args.file)
            produce_measurement_plots(filepath)
        elif args.mode == "beacon":
            filepath = Path(args.file)
            starting_point = [0, 0]
            ending_point = [30, 30]
            produce_beacon_map_plots(filepath,
                                    starting_point, ending_point)
        elif args.mode == "position":
            filepath = Path(args.file)
            produce_position_prediction_plots(filepath)


if __name__ == "__main__":
    main()
