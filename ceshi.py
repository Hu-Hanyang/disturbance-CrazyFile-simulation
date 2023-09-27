import gym
import time
import argparse
import os
import torch
import numpy as np
import warnings
import gym  
from gym.wrappers import Monitor
from PIL import Image
# local imports
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.mpi_tools import is_root_process



env_id='DroneHoverBulletEnvWithAdversary-v0'
env = gym.make(env_id)
# env = Monitor(env, 'test_results_videos', force=True)
frame_width, frame_height = env.render_width, env.render_height
print(frame_width, frame_height)

env.render() 
image = env.capture_image()
print(image.shape)
image = np.asarray(image, dtype=np.uint8)
image = Image.fromarray(image)
image.show()


# for episode in range(10):
#     env.render()
#     observation = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # Random action, replace this with your policy
#         observation, reward, done, _ = env.step(action)

# env.close()

# images = [ [] for _ in range(5)]
# print(len(images))
# print(images)
