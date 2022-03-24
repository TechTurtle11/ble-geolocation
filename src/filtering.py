from abc import ABC
import numpy as np


class BaseFilter(ABC):


    def predict_and_update(self, new_observation):
        pass


    def filter_list(self,array,used_start_as_mean=True):
        i = 0
        if used_start_as_mean:
            i = 1
        filtered_values = [self.predict_and_update(observation) for observation in array[i:]]

        return filtered_values

class KalmanFilter(BaseFilter):


    def __init__(self,previous_mean, previous_var=0, A=1, H=1, Q=1.6, R=6) -> None:
        """
            
        Kalman Filter Implementation
        Parameters:
        previous_mean: previous mean state
        previous_var : previous variance state
        A: The transition constant
        H: measurement constant
        Q: The covariance constant
        R: measurement covariance constant"""
        
        self.previous_mean = previous_mean
        self.previous_var  = previous_var
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R


    def predict_and_update(self,new_observation):
        """
        Prediction and update in Kalman filter
        
        Parameters:
        new_observation: current observation

        Returns:
        new_mean: mean state prediction
        new_var: variance state prediction
        """
        x_mean = self.A * self.previous_mean + np.random.normal(0, self.Q, 1)
        P_mean = self.A * self.previous_var * self.A + self.Q

        K = P_mean * self.H * (1 / (self.H * P_mean * self.H + self.R))
        new_mean = x_mean + K * (new_observation - self.H * x_mean)
        new_var = (1 - K * self.H) * P_mean

        self.previous_mean = new_mean
        self.previous_var = new_var

        return new_mean


class BasicFilter(BaseFilter):

    def __init__(self,previous_observation) -> None:
        self.previous_observation = previous_observation

    def predict_and_update(self,new_observation):

        predicted_observation = new_observation * 0.25 + self.previous_observation * 0.75
        self.previous_observation = predicted_observation
        return predicted_observation


def cheap_filter_list(array, start_mean=None):
    i = 0
    if start_mean is None:
        previous_observation = array[0]
        i += 1
    else:
        previous_observation = start_mean

    filtered_means = np.array([previous_observation])

    while i < len(array):
        filtered_rssi = array[i] * 0.25 + previous_observation * 0.75
        filtered_means = np.append(filtered_means, [filtered_rssi])

        previous_observation = filtered_rssi
        i += 1

    return filtered_means



def filter_list(array, start_mean=None, previous_var=1):
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
        i += 1
    else:
        previous_observation = start_mean

    filtered_means = np.array([previous_observation])
    previous_var = 0
    while i < len(array):
        filtered_rssi, next_covariance = kalman_block(
            previous_observation, previous_var, array[i], A=1, H=1, Q=0.008, R=1)
        filtered_means = np.append(filtered_means, [filtered_rssi])

        previous_observation = filtered_rssi
        previous_var = next_covariance
        i += 1

    return filtered_means