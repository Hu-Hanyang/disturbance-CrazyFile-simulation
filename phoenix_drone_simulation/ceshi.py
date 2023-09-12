import gym
import time
import argparse
import os
import torch
import numpy as np
import warnings

# local imports
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.mpi_tools import is_root_process



env_id="DroneHoverBulletEnvWithAdversary-v0"
env = gym.make(env_id)
print(f"The env is {env}")
# env.render(mode='rgb_array') 
# # print(sth)

# while True:
#     obs = env.reset()
#     done = False
#     while not done:

#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         # print(obs.shape)  # (34,)
#         time.sleep(0.05)
#         if done:
#             obs = env.reset()

# current_path = os.path.dirname(os.path.abspath(__file__))
# father_path = os.path.dirname(current_path)
# print(father_path)