from abc import ABC
import numpy as np


class BaseFilter(ABC):

    def predict_and_update(self, new_observation):
        pass

    def filter_list(self, array):
        i = 0
        filtered_values = [self.predict_and_update(
            observation) for observation in array]

        return filtered_values


class KalmanFilter(BaseFilter):
    def __init__(self, A=1, H=1, Q=1.6, R=6) -> None:
        """

        Kalman Filter Implementation
        Parameters:
        previous_mean: previous mean state
        previous_var : previous variance state
        A: The transition constant
        H: measurement constant
        Q: The covariance constant
        R: measurement covariance constant"""

        self.previous_mean = None
        self.previous_var = 0
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def predict_and_update(self, new_observation):
        """
        Prediction and update in Kalman filter

        Parameters:
        new_observation: current observation

        Returns:
        new_mean: mean state prediction
        new_var: variance state prediction
        """

        if self.previous_mean is None:
            self.previous_mean = new_observation
            return new_observation

        x_mean = self.A * self.previous_mean + np.random.normal(0, self.Q, 1)
        P_mean = self.A * self.previous_var * self.A + self.Q

        K = P_mean * self.H * (1 / (self.H * P_mean * self.H + self.R))
        new_mean = x_mean + K * (new_observation - self.H * x_mean)
        new_var = (1 - K * self.H) * P_mean

        self.previous_mean = new_mean
        self.previous_var = new_var

        return new_mean[0]


class BasicFilter(BaseFilter):

    def __init__(self) -> None:
        self.previous_observation = None

    def predict_and_update(self, new_observation):
        if self.previous_observation is None:
            self.previous_observation = new_observation
            return new_observation

        predicted_observation = new_observation * \
            0.1 + self.previous_observation * 0.9
        self.previous_observation = predicted_observation
        return predicted_observation


class MovingMeanFilter(BaseFilter):
    def __init__(self) -> None:
        self.n = 10
        self.buffer = []

    def predict_and_update(self, new_observation):
        self.buffer.append(new_observation)

        prediction = np.mean(self.buffer)

        if len(self.buffer) >= self.n:
            self.buffer.pop(0)

        return prediction


class MovingMedianFilter(BaseFilter):
    def __init__(self) -> None:
        self.n = 10
        self.buffer = []

    def predict_and_update(self, new_observation):
        self.buffer.append(new_observation)

        prediction = np.median(self.buffer)

        if len(self.buffer) >= self.n:
            self.buffer.pop(0)

        return prediction
