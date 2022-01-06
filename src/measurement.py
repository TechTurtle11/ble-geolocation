from os import path
from bluepy.btle import Scanner, DefaultDelegate
import numpy as np
import math
import logging
from pathlib import Path

logging.basicConfig(filename='logs/measurement.log', level=logging.ERROR)


BEACON_WINDOW = 1 #Seconds
BEACON_SAMPLES_PER_WINDOW = 10 

class ScanDelegate(DefaultDelegate):

    entries={}
    def __init__(self):
        DefaultDelegate.__init__(self)


    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            logging.debug(f"Discovered device address: {dev.addr} rssi: {dev.rssi}")
        elif isNewData:
            logging.debug(f"received new data device address: {dev.addr} rssi: {dev.rssi}")

        device_entries = self.entries.get(dev.addr)
        if device_entries is None:
            self.entries[dev.addr] = np.array([dev.rssi])
        else:
            np.append(self.entries[dev.addr],[dev.rssi])




def get_live_measurement(beacons):
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)

    measurement_general = dict(filter(lambda val: val[0] in beacons))
    measurement_averaged = {k: np.mean(v) for k, v in measurement_general.items()}
    return measurement_averaged
    

def get_training_measurement():
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)

    measurement_general = dict(filter(lambda val: len(val[1]) == BEACON_SAMPLES_PER_WINDOW))
    measurement_averaged = {k: np.mean(v) for k, v in measurement_general.items()}
    return measurement_averaged





def load_training_data(filepath: Path):
    """
    loads the training data for a given Path object
  
  
    Parameters:
    filepath (Path): Filepath containing the training data
  
    Returns:
    dict: Dictionary of beacon_address to training data. Where the training data for each beacon consists of a numpy array of (point,rssi) pairs.
   
    """


    beacons = {} # dictionary with a numpy array of training data for each beacon
    with open(filepath,"r") as file:
        for entry in file.readlines():
            raw_position, measurements = entry.strip("\n").strip("\t").split("&")
            measurement_pairs = [measurement.split(",") for measurement in measurements.split(";")]

            position =  np.array([float(coord) for coord in position.split(",")])

            for address, rssi in measurement_pairs:
                if not address in beacons.keys():
                    beacons[address] = np.array([])
                np.append(beacons[address],np.array([rssi,position]))

    return beacons


def load_measurement(filepath:Path):
    with open(filepath,"r") as file:
        entry = file.readline()
        measurement_pairs = [measurement.split(",") for measurement in entry.split(";")]
        return dict(measurement_pairs)


def write_training_data_to_file(training_data: dict, filepath: Path,mode= "w"):
    with open(filepath,mode) as file:

        for beacon, data in training_data.items():
            lines = [f"{position[0]},{position[1]}&{beacon},{rssi}\n" for position,rssi in data]
            file.writelines(lines)





def collect_training_data():


    training_data = {}

    while True:
        x = input("Enter current x coordinate")
        y = input("Enter current y coordinate")
        position = np.array([x,y])
        print("Computing rssi vector for {position} :")
        measurement = get_training_measurement()

        for beacon,rssi  in measurement.items():
            if not beacon in training_data.keys():
                training_data[beacon] = np.array([])
                np.append(training_data[beacon],np.array([rssi,position]))

        loop_continue = input("Type stop if you have finished collecting training data")
        if "stop" in loop_continue.lower():
            break

    return training_data



def main():


    data = collect_training_data()

    training_data_filepath = Path("data/test_training.txt")
    write_training_data_to_file(data,training_data_filepath)




if __name__ == "__main__":
    main()