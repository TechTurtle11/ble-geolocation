import numpy as np
import abc


class BaseFilter(abc.ABC):

    @abc.abstractmethod
    def predict_and_update(self, new_observation):
        pass

    def filter_list(self, array):
        """Filters a list of values using the current filter"""

        filtered_values = [self.predict_and_update(
            observation) for observation in array]

        return filtered_values


class KalmanFilter(BaseFilter):
    def __init__(self, A=1, B=1, C=1, Q=4, R=0.008) -> None:
        """

        First Order Kalman Filter Implementation
        Parameters:
        previous_mean: previous mean state
        previous_var : previous variance state
        A: state vector
        B: control vector
        C: control vector
        Q: measurement noise
        R: process noise """

        self.previous_mean = None
        self.previous_var = None
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        np.random.seed(3)  # to ensure constant results for evaluation

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
            self.previous_mean = (1 / self.C) * new_observation
            self.previous_var = (1 / self.C) * self.Q * (1 / self.C)
            return new_observation
        else:
            x_mean = self.A * self.previous_mean + self.B * np.random.normal(0, self.R, 1)
            P_mean = (self.A * self.previous_var * self.A) + self.R

            # kalman gain
            K = P_mean * self.C * (1 / ((self.C * P_mean * self.C) + self.Q))

            new_mean = x_mean + K * (new_observation - (self.C * x_mean))
            new_var = P_mean - (K * self.C * P_mean)

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
