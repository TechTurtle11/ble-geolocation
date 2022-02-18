import csv
from pathlib import Path

import numpy as np
import general_helper as gh
from map import Cell


def write_cells_to_file(cells,filepath):
    lines = [f"{cell.center},{cell.probability},{cell.covariance}\n" for cell in cells]
    with open(filepath,"w") as file:
        file.writelines(lines)

def write_position_prediction_to_file(predictions,filepath):
    lines = []

    for position, cells in predictions.values():
        lines.extend([f"{position[0]};{position[1]},{cell.center[0]};{cell.center[1]},{cell.probability},{cell.covariance}\n" for cell in cells])

    with open(filepath,"w") as file:
        file.writelines(lines)


def read_position_prediction_from_file(filepath):
    predictions = {}
    with open(filepath,"r") as file:

        for line in file.readlines():
            parts = line.strip("\n").strip("\t").split(",")

            measured_position = np.array([float(part) for part in parts[0].split(";")])
            predicted_position = np.array([float(part) for part in parts[1].split(";")])
            cell = Cell(predicted_position)      
            cell.probability = float(parts[2])
            cell.covariance = float(parts[3])

            meas_pos_hash = gh.hash_2D_coordinate(*measured_position)
            if not meas_pos_hash in predictions:
                predictions[meas_pos_hash] = [measured_position,[]]

            predictions[meas_pos_hash][1].append(cell)

    return predictions


 
def load_training_data(filepath: Path,windows = False):
    """
    loads the training data for a given Path object
  
  
    Parameters:
    filepath (Path): Filepath containing the training data
  
    Returns:
    dict: Dictionary of beacon_address to training data. Where the training data for each beacon consists of a numpy array of (point,rssi) pairs.
   
    """


    training_data = {} # dictionary with a numpy array of training data for each beacon
    with open(filepath,"r") as file:
        for entry in file.readlines():
            print(entry)
            raw_position, measurements = entry.strip("\n").strip("\t").split("&")

            position =  np.array([float(coord) for coord in raw_position.split(",")])

            beacon,rssi_string = measurements.split(",")
            rssi_strings = rssi_string.split(";")
            if windows: 
                rssi_values = np.array([float(rssi) for rssi in rssi_strings])
                row = [rssi_values,position]
                if not beacon in training_data.keys():
                        training_data[beacon] = np.array([row])
                else:
                    training_data[beacon] = np.append(training_data[beacon],[row],axis=0)
            else:
                for rssi in rssi_strings:
                    row = np.array([float(rssi),*position])
                    if not beacon in training_data.keys():
                        training_data[beacon] = np.array([row])
                    else:
                        training_data[beacon] = np.append(training_data[beacon],[row],axis=0)

    return training_data


def write_training_data_to_file(training_data: dict, filepath: Path,mode= "w"):
    with open(filepath,mode) as file:
        lines = []
        for beacon, data in training_data.items():
            
            for position,rssi_values in data:

                rssi_string = ";".join([str(rssi) for rssi in rssi_values])
                lines.append(f"{position[0]},{position[1]}&{beacon},{rssi_string}\n")

        file.writelines(lines)



def read_measurement_from_file(filepath):
    measurement = []
    with open(filepath,"r") as csv_file:

        for window in csv_file.readlines():
            readings_strings = window.strip("\n").strip("\t").split(",")
            reading = [int(num, base=10) for num in readings_strings]
            measurement.append(reading)

    return measurement


def write_timed_measurement(filepath,readings):
    with open(filepath,"w+") as csv_file:
        csvWriter = csv.writer(csv_file,delimiter=',')
        csvWriter.writerows(readings)