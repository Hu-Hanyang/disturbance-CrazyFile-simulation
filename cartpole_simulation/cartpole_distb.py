import os
import numpy as np
import pybullet as pb
import pybullet_data
# import gym
import gymnasium as gym
from pybullet_utils import bullet_client
import abc
from PIL import Image
from datetime import datetime

# print(gym.__version__)
env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)
print(env.action_space.sample())