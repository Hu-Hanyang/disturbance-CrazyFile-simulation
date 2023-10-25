r""""Environment utilities for Project Phoenix.

Author:     Sven Gronauer, Hanyang Hu
Created:    17.05.2021

Updates:
    14.07.2021: Added low-pass filter and sensor noise (according to USC)
    05.08.2021: Added transformation function for quaternions, rotation matrices
    14.04.2022: Add depreciation warnings for future clean-ups
    11th, October, 2023: Add function 'Boltzmann' by Hanyang
"""
import os
import numpy as np
from math import exp

def Uniform(total_epoch, current_epoch):
    distb_levels = np.arange(low=0.0, high=2.1, accuracy=0.1)
    total_numbers = len(distb_levels)
    interval = total_epoch // total_numbers
    remainder = total_epoch % total_numbers
    if current_epoch < total_epoch - remainder:
        distb_level = np.around((current_epoch // interval), 1)
    else:
        distb_level = np.random.choice(distb_levels)
    return distb_level

def Boltzmann(low=0.0, high=2.1, accuracy=0.1):
    energies = np.array(np.arange(low, high, accuracy))  # Example energy levels
    beta = 1.0  # Inverse temperature (1/kT)

    # Calculate Boltzmann weights
    weights = np.exp(-beta * energies)

    # Normalize to get probabilities
    probabilities = weights / np.sum(weights)

    # Generate random samples from the Boltzmann distribution
    random_state = np.around(np.random.choice(energies, p=probabilities), 1)  
    return random_state


def rad2deg(x):
    r"""Converts radians to degrees."""
    return 180*x/np.pi


def deg2rad(x):
    r"""Converts degrees to radians."""
    return np.pi*x/180


def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'assets')
    return data_path


def get_quaternion_from_euler(rpy) -> np.ndarray:
    r"""Get quaternion [x, y, z, w] from Euler angles RPY [rad].

    See:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    assert rpy.shape == (3, )
    roll, pitch, yaw = rpy
    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = np.cos(halfYaw)
    sinYaw = np.sin(halfYaw)
    cosPitch = np.cos(halfPitch)
    sinPitch = np.sin(halfPitch)
    cosRoll = np.cos(halfRoll)
    sinRoll = np.sin(halfRoll)

    x = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
    y = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
    z = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw
    w = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw

    # Note: PyBullet uses the order: [X,Y,Z,W]
    return np.array([x, y, z, w])


class LowPassFilter:
    def __init__(self, gain: float, time_constant: float, sample_time: float):
        """

        Parameters
        ----------
        gain: K
        time_constant: T
            The filter time constant. Remember: ~4T until step response
        sample_time: T_s
            Typically 1/sim_freq
        """
        self._x = 0
        self.K = gain
        self.T = time_constant
        self.T_s = sample_time

    def apply(self, u):
        ratio = self.T_s / self.T
        self._x = (1 - ratio) * self._x + self.K * ratio * u
        return self._x

    def set(self, x):
        self._x = x


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
