from random import sample
from bluepy.btle import Scanner, DefaultDelegate
import numpy as np
import math
import logging
from pathlib import Path
import math
import csv

logging.basicConfig(filename='logs/measurement.log', level=logging.DEBUG)


BEACON_WINDOW = 1 #Seconds
BEACON_SAMPLES_PER_WINDOW = 10
BEACON_MAC_ADDRESSES = ['e4:5f:01:63:71:64','e4:5f:01:63:71:e5','e4:5f:01:63:71:55','e4:5f:01:63:71:b5','e4:5f:01:63:71:a3']

class ScanDelegate(DefaultDelegate):


    def __init__(self,time_stamps= False):
        DefaultDelegate.__init__(self)
        self._time_stamps = time_stamps
        self.entries={}

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



def get_live_measurement(training_data, previous_measurement = None):
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)

    print(delegate.entries)

    final_measurement = {address:(np.mean(readings),0) for address,readings in delegate.entries.items() if address in BEACON_MAC_ADDRESSES}

    if previous_measurement is not None:
        for beacon,reading in final_measurement.items():
            if beacon in previous_measurement.keys():
                #kalman filter on rssi values
                rssi = reading[0]
                previous_rssi, previous_covariance = previous_measurement[beacon]
                filtered_rssi, next_covariance = kalman_block(previous_rssi,previous_covariance,rssi, A=1, H=1, Q=1.6, R=6)
                final_measurement[beacon] = (filtered_rssi,next_covariance)

        
    return final_measurement
    

def get_training_measurement():
    delegate= ScanDelegate()
    scanner = Scanner().withDelegate(delegate)
    
    for i in range(BEACON_SAMPLES_PER_WINDOW):
        devices = scanner.scan(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW)


    measurement_general = {address:readings for address,readings in delegate.entries.items() if address in BEACON_MAC_ADDRESSES}
    measurement_averaged = {k: np.mean(v) for k, v in measurement_general.items()}
    return measurement_averaged


def timed_measurement(beacon_name, time):

    total_windows = math.ceil(time/BEACON_WINDOW)
    total_beacon_samples = total_windows*BEACON_SAMPLES_PER_WINDOW

    raw_readings = []

    for window in range(total_windows):
        delegate= ScanDelegate()
        scanner = Scanner().withDelegate(delegate)
        for _ in range(BEACON_SAMPLES_PER_WINDOW):
            devices = scanner.scan(1.1*(BEACON_WINDOW/BEACON_SAMPLES_PER_WINDOW))

        if beacon_name in delegate.entries.keys():
            window_readings = delegate.entries[beacon_name]#raw rssi data
            raw_readings.append(window_readings)

    return raw_readings

def collect_and_write_timed_measurement(beacon_name,time,filepath):
    readings = timed_measurement(beacon_name,time)
    print(readings)
    with open(filepath,"w+") as csv_file:
        csvWriter = csv.writer(csv_file,delimiter=',')
        csvWriter.writerows(readings)

def read_measurement_from_file(filepath):
    measurement = []
    with open(filepath,"r") as csv_file:

        for window in csv_file.readlines():
            readings_strings = window.strip("\n").strip("\t").split(",")
            reading = [int(num, base=10) for num in readings_strings]
            measurement.append(reading)

    return measurement

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


def kalman_block(previous_mean, previous_var, new_observation, A, H, Q, R):
    """
    Prediction and update in Kalman filter

    Parameters:
    previous_mean: previous mean state
    previous_var : previous variance state
    new_observation: current observation
    A: The transition constant
    H: measurement constant
    Q: The covariance constant
    R: measurement covariance constant

    Returns:
    new_mean: mean state prediction
    new_var: variance state prediction
    """

    # https://arxiv.org/abs/1204.0375v1 for reference

    x_mean = A * previous_mean + np.random.normal(0, Q, 1)
    P_mean = A * previous_var * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    new_mean = x_mean + K * (new_observation - H * x_mean)
    new_var = (1 - K * H) * P_mean

    return new_mean, new_var


def filter_list(array, start_mean=None,previous_var=1):
    """
    filters list using a kalman filter
    parameters are setup for rssi values

    Parameters:
    array: the array to be filtered
    start_mean: the start start if wanted to adjust
    start_var: the start variance if wanted to adjust

    Returns:
    filtered_means: filtered array
    """
    i = 0
    if start_mean is None:
        previous_observation = array[0]
        i+=1
    else:
        previous_observation = start_mean
    
    filtered_means = np.array([previous_observation])
    previous_var = 0
    while i < len(array):
        filtered_rssi, next_covariance = kalman_block(previous_observation, previous_var, array[i], A=1, H=1, Q=0.008, R=1)
        filtered_means = np.append(filtered_means,[filtered_rssi])

        previous_observation = filtered_rssi
        previous_var = next_covariance
        i+=1
    
    return filtered_means


def main():
    

    data = collect_training_data()

    training_data_filepath = Path("data/intel_indoor_training.txt")
    print(data)
    write_training_data_to_file(data,training_data_filepath)




if __name__ == "__main__":

    measurement_filepath = Path("data/test_measurement.csv")
    collect_and_write_timed_measurement('e4:5f:01:63:71:64',1000,measurement_filepath)
    print(read_measurement_from_file(measurement_filepath))
    main()