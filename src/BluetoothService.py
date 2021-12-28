import logging
from bluepy.btle import Scanner, DefaultDelegate
import numpy as np
import scipy

BEACON_PREFIX = "geolocation_beacon"


class ScanDelegate(DefaultDelegate):

    entries={}
    def __init__(self):
        DefaultDelegate.__init__(self)


    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            logging.debug(f"Discovered device address: {dev.addr} rssi: {dev.rssi}")
        elif isNewData:
            logging.debug(f"received new data device address: {dev.addr} rssi: {dev.rssi}")

        devcice_entries = self.entries.get(dev.addr)
        if devcice_entries is None:
            self.entries[dev.addr] = [dev.rssi]
        else:
            self.entries[dev.addr].append(dev.rssi)


def create_beacon_survey_maps(measurements: dict):

    #p_distance = lambda X: np.product([scipy.stats.norm.pdf(abs(X[i+1]-X[i]), 0.1*1.5, 0.1*0.2) for i in range(len(X)-1)])
    p_orientation = lambda X: np.product([])

    for beacon,observations in measurements.items():
        print(beacon)
        print(observations)


def calculate_cell_probabilities(measurements, beacon_survey_maps):
    cell_probabilties = np.zeros((20,20))
    standard_deviation = 1

    for i in range(20):
        for j in range(20):
            position = (i,j)
            distance = 0
            n= len(measurements)
            for beacon,measurement in measurements.items():
                distance+= (measurement - beacon_survey_maps[beacon](position))**2
            distance = distance/n

            p = np.exp(-np.exp2(distance)/(2*np.exp2(standard_deviation)))
            cell_probabilties[i,j] = p


    return cell_probabilties




def main():
    logging.basicConfig(filename='bluetooth.log', level=logging.ERROR)

    for i in range(1):
        delegate= ScanDelegate()
        scanner = Scanner().withDelegate(delegate)
        
        for i in range(10):
            devices = scanner.scan(0.1)

        print(delegate.entries)
        """

        # generate map with entries
        survey_maps = create_beacon_survey_maps(delegate.entries)
        average_measurements = {beacon:np.mean(measurements) for beacon,measurements in delegate.entries.items()}

        cell_probabilties = np.zeros((20,20))
        cell_probabilties = calculate_cell_probabilities(average_measurements,survey_maps)
        most_likely_cell = np.argmax(cell_probabilties)"""
        
if __name__ == "__main__":
    main()