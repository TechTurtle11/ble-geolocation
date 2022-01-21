from os import path
from bluepy.btle import Scanner, DefaultDelegate
from matplotlib.pyplot import axis
import numpy as np
import math
import logging
from pathlib import Path

logging.basicConfig(filename='logs/measurement.log', level=logging.DEBUG)


BEACON_WINDOW = 1 #Seconds
BEACON_SAMPLES_PER_WINDOW = 10
BEACON_MAC_ADDRESSES = ['e4:5f:01:63:71:64','e4:5f:01:63:71:e5','e4:5f:01:63:71:55','e4:5f:01:63:71:b5','e4:5f:01:63:71:a3']

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
            self.entries[dev.addr] = [dev.rssi]
        else:
            self.entries[dev.addr].append(dev.rssi)




def get_live_measurement(training_data):
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)

    measurement_general = {address:readings for address,readings in delegate.entries.items() if address in BEACON_MAC_ADDRESSES}
    measurement_averaged = {k: np.mean(v) for k, v in measurement_general.items()}
    return measurement_averaged
    

def get_training_measurement():
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)

    print(delegate.entries)

    measurement_general = {address:readings for address,readings in delegate.entries.items() if address in BEACON_MAC_ADDRESSES}
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


    training_data = {} # dictionary with a numpy array of training data for each beacon
    with open(filepath,"r") as file:
        for entry in file.readlines():
            raw_position, measurements = entry.strip("\n").strip("\t").split("&")
            measurement_pairs = [measurement.split(",") for measurement in measurements.split(";")]

            position =  np.array([float(coord) for coord in raw_position.split(",")])

            for beacon, rssi in measurement_pairs:
                row = np.array([float(rssi),*position])
                if not beacon in training_data.keys():
                    training_data[beacon] = np.array([row])
                else:
                    training_data[beacon] = np.append(training_data[beacon],[row],axis=0)

    return training_data



def write_training_data_to_file(training_data: dict, filepath: Path,mode= "w"):
    with open(filepath,mode) as file:

        for beacon, data in training_data.items():
            lines = [f"{position[0]},{position[1]}&{beacon},{rssi}\n" for position,rssi in data]
            file.writelines(lines)





def collect_training_data():

    training_data = {}
    get_training_measurement()

    while True:
        x = input("Enter current x coordinate: ")
        y = input("Enter current y coordinate: ")
        position = np.array([x,y])
        print(f"Computing rssi vector for {position} :")
        measurement = get_training_measurement()
        print(f"Measurement was {measurement} :")

        for beacon,rssi  in measurement.items():

            if not beacon in training_data.keys():
                training_data[beacon] = []
            training_data[beacon].append([position,rssi])


        loop_continue = input("Type stop if you have finished collecting training data: ")
        if "stop" in loop_continue.lower():
            break

    return training_data



def main():

    data = collect_training_data()

    training_data_filepath = Path("data/cl_indoor_training.txt")
    print(data)
    write_training_data_to_file(data,training_data_filepath)




if __name__ == "__main__":
    main()