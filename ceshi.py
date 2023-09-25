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

image = env.render(mode='rgb_array') 
print(image.shape)
image = np.asarray(image, dtype=np.uint8)
image = Image.fromarray(image)
image.show()
print(image.shape)


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

import imageio
import numpy as np

# Assuming 'frames' is your list of numpy ndarrays
# Make sure 'frames' contains 50 ndarrays of shape (240, 320, 3)

# Define the filename for the output video
output_file = 'output_video.mp4'

# Create a writer object
writer = imageio.get_writer(output_file, fps=30)  # You can adjust the frames per second (fps)

# Iterate through the frames and add them to the video
for frame in frames:
    writer.append_data(frame)

# Close the writer
writer.close()

print(f'Video saved as {output_file}')

