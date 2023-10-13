import gym
import time
import argparse
import os
import torch
import numpy as np
import warnings
import gym  
from gym.wrappers import Monitor
# from PIL import Image
# local imports
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.mpi_tools import is_root_process
import numpy as np
from scipy.stats import boltzmann
import matplotlib.pyplot as plt



# def Boltzmann(N):
#     lambda_ = 100
#     boltzmann = np.zeros(N)
#     for n in range(N):
#         boltzmann[n] = (1-np.exp(-lambda_))*np.exp(-lambda_*n)/(1-np.exp(-lambda_*N)) 
#     return 2 * boltzmann

# results = Boltzmann(100)
# print(results[0])
# plt.figure()
# plt.scatter(range(100), results)
# plt.show()


# env_id='DroneHoverBulletEnvWithAdversary-v0'
# env = gym.make(env_id)
# # env = Monitor(env, 'test_results_videos', force=True)
# frame_width, frame_height = env.render_width, env.render_height
# print(frame_width, frame_height)

# env.render() 
# image = env.capture_image()
# print(image.shape)
# image = np.asarray(image, dtype=np.uint8)
# image = Image.fromarray(image)
# image.show()


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


# print(f"The trained env is xx; \n The test env is xxx")

# Define energy levels and corresponding probabilities
def Boltzmann1():
    energies = np.array(np.arange(0.0, 2.1, 0.1))  # Example energy levels
    beta = 1.0  # Inverse temperature (1/kT)

    # Calculate Boltzmann weights
    weights = np.exp(-beta * energies)

    # Normalize to get probabilities
    probabilities = weights / np.sum(weights)

    # Generate random samples from the Boltzmann distribution
    random_state = np.random.choice(energies, p=probabilities)  
    return random_state

# print(f"Randomly selected state: {random_state}")


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

N = 10000
result = np.zeros(N)
for i in range(N):
    # result[i] = Boltzmann1()
    result[i] = Boltzmann()

# plt.figure()
# plt.scatter(range(N), result)
# # plt.plot(result)
# plt.show()


plt.hist(np.around(result, 1), bins=210, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Data')

# Show the plot
plt.show()

# a = np.array(np.arange(0.00, 2.10, 0.01))
# print(a)
# print(np.around(a,1))