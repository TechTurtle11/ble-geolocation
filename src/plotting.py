from __future__ import annotations
import decimal
import matplotlib.pyplot as plt
import numpy as np
from map import Map
from constants import MapAttribute


def plot_beacon_map_rssi(beacon_name, beacon_map, starting_point=[-10, -10], ending_point=[20, 20]):

    x_samples = np.arange(starting_point[0], ending_point[0], 1)
    y_samples = np.arange(starting_point[0], ending_point[0], 1)[::-1]

    predictions = [[beacon_map.predict([np.array([x, y])])[
        0] for x in x_samples] for y in y_samples]

    predictions = np.rint(predictions).astype(int)

    plot_heatmap(x_samples, y_samples, predictions,
                 f"RSSI Map for beacon: {beacon_name}", annotations=True, colorbar=True)


def plot_beacon_map_covariance(beacon_name, beacon_map, starting_point=[-10, -10], ending_point=[20, 20]):
    x_samples = np.arange(starting_point[0], ending_point[0], 1)
    y_samples = np.arange(starting_point[0], ending_point[0], 1)[::-1]
    predictions = [[beacon_map.predict([np.array([x, y])], return_cov=True)[
        1][0][0] for x in x_samples] for y in y_samples]

    plot_heatmap(x_samples, y_samples, np.array(predictions),
                 f"Covariance Map for beacon: {beacon_name}", colorbar=True)


def plot_heatmap(x_samples, y_samples, predictions, title, annotations=False, colorbar=False):

    fig, ax = plt.subplots()
    im = ax.imshow(predictions)

    # Setting the labels
    ax.set_xticks(np.arange(len(y_samples)))
    ax.set_yticks(np.arange(len(x_samples)))

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
        raise ValueError(f"{attribute} isnt supported")

    attribute_values = np.array(attribute_values).reshape(
        (len(x_samples), len(y_samples)))

    plot_heatmap(x_samples, y_samples, attribute_values, title)
