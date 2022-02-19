import logging
import math

import numpy as np

import measurement as meas


def load_map(filename: str):
    beacons = {}
    positions = {}
    points = []
    with open(filename, "r") as file:
        for entry in file.readlines():
            beacon_address, position, measurements = entry.strip(
                "\n").strip("\t").split("&")
            measurement_pairs = [measurement.split(
                ",") for measurement in measurements.split(";")]
            if not beacon_address in beacons.keys():
                beacons[beacon_address] = []
            beacons[beacon_address].append(measurement_pairs)

            positions[beacon_address] = np.array(
                [float(coord) for coord in position.split(",")])
    return beacons, positions


def knn(beacons, positions, observed_measurement, k=3):
    distances = []

    for beacon, training_measurements in beacons.items():
        distance = math.sqrt(sum([(observed_measurement[address] - training_measurements[address])
                                  ** 2 for address in training_measurements.keys()]) / len(beacons))
        distances.append((beacon, distance))

    sorted_distances = sorted(beacons, key=lambda x: x[1], reverse=True)

    first_k = sorted_distances[:k]
    position = np.zeros(2)

    distance_sum = sum([distance for _, distance in first_k])
    for closest_beacon, distance in first_k:
        position += (distance / distance_sum) * positions[closest_beacon]

    print(f"position coordinate: {position}")
    return position


def get_measurement_covariance(old, new):
    signal_variance = 1
    length_scale = 1

    covariance = {beacon: signal_variance ** 2 * np.exp((-1 / (2 * np.exp2(length_scale))) * np.exp2(
        np.absolute(old[beacon] - new[beacon]))) for beacon in old.keys()}
    return covariance


def main():
    logging.basicConfig(filename='main.log',
                        level=logging.ERROR)

    k = 3

    beacons, positions = load_map("testfile.txt")

    current_measurement = meas.get_measurement(beacons.keys())
    current_position = knn(beacons, positions, current_measurement)

    while True:
        # assume .5 meter move
        predicted_position = np.array(
            [np.random.normal(loc=value, scale=0.5, size=1) for value in current_position])

        measurement = meas.get_measurement(beacons.keys())
        k_means_position = knn(beacons, positions, measurement)
        signal_variance = 1
        length_scale = 1
        covariance = signal_variance ** 2 * \
                     np.exp((-1 / (2 * np.exp2(length_scale))) *
                            np.exp2(np.absolute(k_means_position - current_position)))


if __name__ == "__main__":
    main()
