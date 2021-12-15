import logging


class Beacon:

    def __init__(self, serial_number, location):
        self._serial_number = serial_number
        self._rssi_map = {}
        self._location = np.array(location)  # assumes 2 or 3 length

    def add_reading(self, point, rssi_reading):
        self._rssi_map[point] = rssi_reading

    def get_reading(self, point):
        return self._rssi_map[point]

    @property
    def get_rssi_map(self):
        return self._rssi_map

    @property
    def get_locstion(self):
        return self._location


class ReferencePoint:

    def __init__(self, location):
        self._rssi_map = {}
        self._location = np.array(location)  # assumes 2 or 3 length

    def add_reading(self, beacon_id, reading):
        self._rssi_map[beacon_id] = reading

    def get_reading(self, beacon_id):
        if beacon_id in self._rssi_map.keys():
            return self._rssi_map[beacon_id]
        else:
            logging.error("A beacon wasnt found with that id")

    @property
    def get_rssi_map(self):
        return self._rssi_map

    @property
    def get_location(self):
        return self._location


class Map:
    def __init__(self):
        self._beacons = {}
        self._reference_points = {}

    def add_beacon(self, uid, location):
        if uid in self._beacons.keys():
            logging.warning(
                "a beacon already exists with this id overwriting the beacon information")
        self._beacons[uid] = Beacon(uid, location)

    def add_measurement(self, beacon_id, position, reading):
        if beacon_id in self._beacons.keys():
            self._beacons[beacon_id].add_reading(position, reading)
            self._reference_points[position].add_reading(beacon_id, reading)
        else:
            logging.error("A beacon wasnt found with that id")

    @property
    def reference_points(self):
        return self._reference_points


def binary_set(A):
    '''
    get the binary set of A

            Parameters:
                    A (list) list of tuples of beacon_id to rssi values
            Returns:
                the set of beacon tuples 
    '''
    new_set = {}
    set_length = len(A)
    for i in range(set_length):
        for j in range(i + 1, set_length):
            new_set.add((A[i][0], A[j][0]))
    return new_set


def reversions(A, B):
    '''
    get the number of discordant pairs

            Parameters:
                    A (list) list of tuples of beacon_ids
                    B (list) list of tuples of beacon_ids
    '''
    revs = 0

    for _, avg_rssi in A:
        if not (avg_rssi in B):
            revs += 1

    return revs


def predict_position(taus, positions):
    '''
    predicts position of a point using the weighted positions 

            Parameters:
                    taus (list) the weights for each positions
                    positions (list) the cooridinate positions of the beacons
    '''
    tau_sum = sum(taus)
    average_position = sum(
        [constant * position/tau_sum for constant, position in zip(taus, positions)])
    return average_position


def calculate_tau(measured_rssi_readings, reference_point):
    '''
    calculate tau for the measured point relative to  reference point

            Parameters:
                    measured_rssi_readings (dict) the map of beacon ids to rssi values
                    reference_point (Point) the reference point we are comparing with
    '''
    measured_point_readings = len(measured_rssi_readings.keys())
    measured_rssi_readings = sorted(
        measured_rssi_readings.items(), key=lambda item: item[1], reverse=True)
    point_rssi_readings = sorted(
        reference_point.rssi_map.items(), key=lambda item: item[1], reverse=True)

    measured_b_set = binary_set(measured_rssi_readings)
    point_b_set = binary_set(point_rssi_readings)
    revs = reversions(measured_b_set, point_b_set)

    tau = 1 - 2 * revs / (len(measured_point_readings) *
                          (len(measured_point_readings) - 1))
    return tau


def main():
    logging.basicConfig(filename='main.log',
                        level=logging.ERROR)

    reference_points = 4
    number_beacons = 5
    k = 3
    # todo build map
    area_map = Map()
    # todo readings for point must be obtained
    measured_rssi_readings = {}
    reference_points = area_map.reference_points
    tau_map = []
    for ref_point in reference_points:
        tau_map.append((ref_point.location, calculate_tau(
            measured_rssi_readings, ref_point)))

    # larger tau suggests closer
    sorted_taus = sorted(tau_map, key=lambda item: item[1], reverse=True)

    # find k nearest reference points
    nearest_points = sorted_taus[:k]
    # position is calculated using average distance between nearest reference points based on tau values
    position = predict_position(*nearest_points)


if __name__ == "__main__":
    main()
