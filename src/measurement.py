import logging
import math
from pathlib import Path

import numpy as np
from bluepy.btle import Scanner, DefaultDelegate

import Utils.constants as const
import Utils.file_helper as fh
from Processing.filtering import KalmanFilter
import Utils.general_helper as gh
import argparse

logging.basicConfig(filename='logs/measurement.log', level=logging.DEBUG)


class ScanDelegate(DefaultDelegate):

    def __init__(self, time_stamps=False):
        DefaultDelegate.__init__(self)
        self._time_stamps = time_stamps
        self.entries = {}

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            logging.debug(
                f"Discovered device address: {dev.addr} rssi: {dev.rssi}")
        elif isNewData:
            logging.debug(
                f"received new data device address: {dev.addr} rssi: {dev.rssi}")

        device_entries = self.entries.get(dev.addr)
        if device_entries is None:
            self.entries[dev.addr] = [dev.rssi]
        else:
            self.entries[dev.addr].append(dev.rssi)


def get_live_measurement(previous_measurement=None, processing=True, measurement_process=const.MeasurementProcess.MEDIAN):
    delegate = ScanDelegate()
    scanner = Scanner().withDelegate(delegate)

    for i in range(const.BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(const.BEACON_WINDOW /
                               const.BEACON_SAMPLES_PER_WINDOW)

    raw_measurement = {address: readings for address, readings in delegate.entries.items(
    ) if address in const.BEACON_MAC_ADDRESSES}

    if processing:
        def processing_function(v): return v
        if measurement_process is const.MeasurementProcess.MEAN:
            processing_function = np.mean
        elif measurement_process is const.MeasurementProcess.MEDIAN:
            processing_function = np.median
        else:
            raise ValueError(
                f"This value {measurement_process} has not been implemented")

        if previous_measurement is None:
            final_measurement = {address: (processing_function(readings), KalmanFilter(processing_function(readings))) for address,
                                 readings in raw_measurement.items()}

        else:
            final_measurement = {}
            for beacon, readings in raw_measurement.items():
                if beacon in previous_measurement.keys():
                    _, filter = previous_measurement[beacon]
                    filtered_rssi = filter.predict_and_update(
                        processing_function(readings))
                    final_measurement[beacon] = (
                        round(filtered_rssi, 3), filter)

    else:
        final_measurement = raw_measurement

    return final_measurement


def get_training_measurement(beacons_addresses):
    delegate = ScanDelegate()
    scanner = Scanner().withDelegate(delegate)

    for j in range(5):
        for i in range(const.BEACON_SAMPLES_PER_WINDOW):
            devices = scanner.scan(0.1)

    measurement_general = {address: readings for address,
                           readings in delegate.entries.items() if address in beacons_addresses}

    return measurement_general


def timed_measurement(beacon_name, time):
    total_windows = math.ceil(time / const.BEACON_WINDOW)
    total_beacon_samples = total_windows * const.BEACON_SAMPLES_PER_WINDOW

    raw_readings = []

    for window in range(total_windows):
        delegate = ScanDelegate()
        scanner = Scanner().withDelegate(delegate)
        for _ in range(const.BEACON_SAMPLES_PER_WINDOW):
            devices = scanner.scan(
                1.1 * (const.BEACON_WINDOW / const.BEACON_SAMPLES_PER_WINDOW))

        if beacon_name in delegate.entries.keys():
            window_readings = delegate.entries[beacon_name]  # raw rssi data
            raw_readings.append(window_readings)

    return raw_readings


def collect_and_write_timed_measurement(beacon_name, time, filepath):
    readings = timed_measurement(beacon_name, time)
    fh.write_timed_measurement(filepath, readings)


def process_training_data(training_data, type=const.MeasurementProcess.MEDIAN):
    """processes windowed training data"""
    processed_training_data = {}
    position_beacon_map = {}  # holds which beacons are used for each position
    hashed_position_map = {}
    for beacon, beacon_data in training_data.items():
        for window_data, position in beacon_data:
            if type is const.MeasurementProcess.MEAN:
                rssi_values = [np.mean(window_data)]
            elif type is const.MeasurementProcess.ALL:
                rssi_values = window_data
            elif type is const.MeasurementProcess.QUANTILE:
                rssi_values = np.quantile(window_data, [0.25, 0.5, 0.75])
            elif type is const.MeasurementProcess.MEDIAN:
                rssi_values = [np.median(window_data)]
            else:
                raise ValueError(f"This value {type} has not been implemented")

            for rssi in rssi_values:
                row = np.array([float(rssi), *position])
                if not beacon in processed_training_data.keys():
                    processed_training_data[beacon] = np.array([row])
                else:
                    processed_training_data[beacon] = np.append(
                        processed_training_data[beacon], [row], axis=0)

            h = gh.hash_2D_coordinate(*position)
            hashed_position_map[h] = position
            if h not in position_beacon_map.keys():
                position_beacon_map[h] = []
            position_beacon_map[h].append(beacon)

    return processed_training_data


def process_evaluation_data(evaluation_data, type=const.MeasurementProcess.MEDIAN):
    """processes windowed evaluation data"""
    processed_evaluation_data = []
    for position, beacon_pairs in evaluation_data:
        processed_beacon_pairs = {}
        for beacon_address, window_data in beacon_pairs.items():
            if type is const.MeasurementProcess.MEAN:
                rssi_value = np.mean(window_data)
            elif type is const.MeasurementProcess.MEDIAN:
                rssi_value = np.median(window_data)
            else:
                raise ValueError(f"This value {type} has not been implemented")

            processed_beacon_pairs[beacon_address] = rssi_value
        processed_evaluation_data.append([position, processed_beacon_pairs])

    return processed_evaluation_data


def collect_evaluation_data():
    evaluation_data = []
    print("Collecting evaluation data: \n")

    while True:
        try:
            x = input("Enter current x coordinate: ")
            y = input("Enter current y coordinate: ")
            position = np.array([float(x), float(y)])
            h = gh.hash_2D_coordinate(*position)

            print(f"Computing rssi vector for {position} :")
            previous_measurement = None
            for i in range(10):
                previous_measurement = get_live_measurement(
                    previous_measurement, processing=False)

                evaluation_data.append([position, previous_measurement])

            loop_continue = input(
                "Type stop if you have finished collecting evaluation data: ")
            if "stop" in loop_continue.lower():
                break
        except ValueError:
            print("please retry with valid coords")

    return evaluation_data


def collect_training_data():
    training_data = {}

    beacon_positions = {}
    for beacon in const.BEACON_MAC_ADDRESSES:
        using = input(f"Enter true if using {beacon}: ")
        if using.lower() == "true":
            x = input(f"Enter current x coordinate for {beacon}: ")
            y = input(f"Enter current y coordinate for {beacon}: ")
            beacon_positions[beacon] = np.array([x, y])

    while True:
        x = input("Enter current x coordinate: ")
        y = input("Enter current y coordinate: ")
        position = np.array([x, y])
        print(f"Computing rssi vector for {position} :")
        measurement = get_training_measurement(list(beacon_positions.keys()))
        print(f"Measurement was {measurement} :")

        for beacon, rssi_values in measurement.items():

            if not beacon in training_data.keys():
                training_data[beacon] = []

            training_data[beacon].append([position, rssi_values])

        loop_continue = input(
            "Type stop if you have finished collecting training data: ")
        if "stop" in loop_continue.lower():
            break

    return beacon_positions, training_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Collect Training data")
    parser.add_argument("file", help="The file to save the results too")
    args = parser.parse_args()

    if args.mode not in ["training", "evaluation", "timed"]:
        print("Mode should be either evaluation or training")

    else:
        filepath = Path(args.file)
        if args.mode == "training":
            beacon_positions, data = collect_training_data()
            fh.write_training_data_to_file(beacon_positions, data, filepath)
        elif args.mode == "evaluation":
            data = collect_evaluation_data()
            fh.write_evaluation_data_to_file(data, filepath)
        elif args.mode == "timed":
            collect_and_write_timed_measurement(
                'e4:5f:01:63:71:64', 60, filepath)


if __name__ == "__main__":
    main()
