from __future__ import annotations
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import numpy as np
from map import Map
from constants import MapAttribute
import measurement as measure
from itertools import chain
from beacon import create_beacons


def plot_beacon_map_rssi(beacon_name, beacon, starting_point=[-10, -10], ending_point=[20, 20],offset=False):

    x_samples = np.arange(starting_point[0], ending_point[0], 1)
    y_samples = np.arange(starting_point[0], ending_point[0], 1)[::-1]

    predictions = [[beacon.predict_rssi(np.array([x, y]),offset=offset) for x in x_samples] for y in y_samples]

    predictions = np.rint(predictions).astype(int)

    fig,ax = plot_heatmap(x_samples, y_samples, predictions,
                 f"RSSI Map for beacon: {beacon_name} at {beacon.position}","Coordinate (m)","Coordinate (m)", annotations=True, colorbar=True)


def plot_beacon_map_covariance(beacon_name, beacon_map, starting_point=[-10, -10], ending_point=[20, 20]):
    x_samples = np.arange(starting_point[0], ending_point[0], 1)
    y_samples = np.arange(starting_point[0], ending_point[0], 1)[::-1]
    predictions = [[beacon_map.predict([np.array([x, y])], return_cov=True)[
        1][0][0] for x in x_samples] for y in y_samples]

    plot_heatmap(x_samples, y_samples, np.array(predictions),
                 f"Covariance Map for beacon: {beacon_name}","Coordinate (m)","Coordinate (m)", colorbar=True)

def plot_training_data(training_data):
    x_samples = []
    y_samples = []
    rssi_values = []
    for beacon_data in training_data.values():
        for point,value in beacon_data:
            x_samples.append(point[0])
            y_samples.append(point)
            rssi_values.append(value)

    


def plot_heatmap(x_samples, y_samples, predictions, title,x_label,y_label, annotations=False, colorbar=False):

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
    plt.show()
    return fig,ax


def plot_map_attribute(map: Map, attribute: MapAttribute):

    cells = map.get_cells
    start, end = map.get_dimensions
    cell_size = map.get_cell_size

    x_samples = np.arange(start=start[0], stop=end[0], step=cell_size)
    y_samples = np.arange(start=start[1], stop=end[1], step=cell_size)[::-1]

    if attribute is MapAttribute.PROB:
        title = f"Probability Heatmap "
        attribute_values = [cell.probability for cell in cells]
    elif attribute is MapAttribute.COV:
        title = f"Covariance Heatmap "
        attribute_values = [cell.covariance for cell in cells]
    else:
        raise ValueError(f"{attribute} isn't supported")

    attribute_values = np.array(attribute_values).reshape(
        (len(x_samples), len(y_samples)))

    fig,ax = plot_heatmap(x_samples, y_samples, attribute_values, title,"Coordinate (m)","Coordinate (m)",)



def plot_rssi_distance(beacon,beacon_location):

    fig, ax = plt.subplots()
    data = beacon.training_data 

    rssi_values = data.T[0]
    distances = np.linalg.norm(data.T[1:].T - beacon_location,axis=1)


    predicted_rssi_values = np.array([beacon.predict_offset_rssi(point) for point in data.T[1:].T])


    plt.xlabel("Distance(m)")
    plt.ylabel("RSSI Value(-DBm)")
    plt.title(f"Plot showing rssi values with distance for {beacon}")

    plt.scatter(distances,rssi_values)
    plt.scatter(distances,predicted_rssi_values)


def plot_rssi_readings_over_time(data_set,title="unknown"):
    plot_comparison(data_set,"Time","RSSI Values(-DBm)",title)

def plot_comparison(data_sets,x_axis,y_axis,title):

    fig, ax = plt.subplots()

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    for name,readings in data_sets.items():
        plt.plot(readings,label = name)

    plt.legend(loc ="lower right")


def plot_filtered_rssi_comparison(measurement,title,round=False):
    if round:
        data_set = {"non-filtered":measurement,"filtered":np.round(measure.filter_list(measurement)),"filtered_cheap":np.round(measure.cheap_filter_list(measurement))}
    else:
        data_set = {"non-filtered":measurement,"filtered":measure.filter_list(measurement),"filtered_cheap":measure.cheap_filter_list(measurement)}
    plot_rssi_readings_over_time(data_set,title)


def produce_measurement_plots(measurement_filepath,round=False):
    
    measurement = measure.read_measurement_from_file(measurement_filepath)
    mean_measurement = np.array([np.mean(window) for window in measurement])
    flattened_readings = np.array(list(chain.from_iterable(measurement)))

    plot_filtered_rssi_comparison(measurement[0],"Window comparison",round)
    plot_filtered_rssi_comparison(mean_measurement,"Mean Kalman Filter comparison",round)
    plot_filtered_rssi_comparison(flattened_readings,"Raw Kalman filter comparison",round)
    plt.show()


def produce_beacon_map_plots(training_data_filepath,starting_point,ending_point):
    training_data = measure.load_training_data(training_data_filepath)
    training_data = measure.process_training_data(training_data)
    beacons = create_beacons(training_data)

    for address, beacon in beacons.items():
        plot_rssi_distance(beacon,beacon.position)
        plot_beacon_map_rssi(address, beacon, starting_point, ending_point,offset=True)
        plot_beacon_map_covariance(address, beacon.get_map, starting_point, ending_point)
        plt.show()

def produce_rotation_plot():
    measurement_filepaths = {angle: Path(f"data/test_rotation_{angle}_measurement.csv") for angle in [0,90,180,270,]}
    measurements = {angle:measure.read_measurement_from_file(filepath) for angle,filepath in measurement_filepaths.items()}
    measurements = {angle:measure.filter_list(np.array(list(chain.from_iterable(measurement)))) for angle,measurement in measurements.items()}
    plot_rssi_readings_over_time(measurements, "RSSI by angle of rotation")

def main():


    #produce_rotation_plot()
    #input()
    measurement_filepath = Path("data/test_measurement.csv")
    produce_measurement_plots(measurement_filepath,round=True)
    input()
    training_data_filepath = Path("data/working_test.txt")
    starting_point = [-2, -2]
    ending_point = [15, 15]
    produce_beacon_map_plots(training_data_filepath,starting_point,ending_point)
    

if __name__ == "__main__":
    print("hello")
    main()