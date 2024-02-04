import gym
import time
import argparse
import os
import torch
import csv
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


# env_id='DroneHoverBulletEnvWithRandomHJAdversary-v0'
# env = gym.make(env_id)
# env = Monitor(env, 'test_results_videos', force=True)
# frame_width, frame_height = env.render_width, env.render_height
# print(frame_width, frame_height)

# env.render() 
# image = env.capture_image()
# print(image.shape)
# image = np.asarray(image, dtype=np.uint8)
# image = Image.fromarray(image)
# image.show()

# observation = env.reset()
# action = env.action_space.sample()  # Random action, replace this with your policy
# # print(f"The action space is: {action}")
# # print(f"The action space shape is: {action.shape}")
# observation, reward, done, _ = env.step(action)

# for episode in range(10):
#     # env.render()
#     observation = env.reset()
#     # print(f"The observation space in the {episode} is: {observation}")
#     print(f"The observation space shape in the {episode} is: {observation.shape}")

#     done = False
#     while not done:
#         action = env.action_space.sample()  # Random action, replace this with your policy
#         # print(f"The action space in the {episode} is: {action}")
#         # print(f"The action space shape is: {action.shape}") 
#         observation, reward, done, _ = env.step(action)
#         print(f"The observation space shape in the {episode} is: {observation.shape}")

#         # print(done)

# env.close()

# images = [ [] for _ in range(5)]
# print(len(images))
# print(images)


# print(f"The trained env is xx; \n The test env is xxx")

# Define energy levels and corresponding probabilities

# print(f"Randomly selected state: {random_state}")

# def Boltzmann(low=0.0, high=2.1, accuracy=0.1):
#     energies = np.array(np.arange(low, high, accuracy))  # Example energy levels
#     beta = 1.0  # Inverse temperature (1/kT)

#     # Calculate Boltzmann weights
#     weights = np.exp(-beta * energies)

#     # Normalize to get probabilities
#     probabilities = weights / np.sum(weights)

#     # Generate random samples from the Boltzmann distribution
#     random_state = np.around(np.random.choice(energies, p=probabilities), 1)  
#     return random_state

# N = 10000
# result = np.zeros(N)
# for i in range(N):
#     # result[i] = Boltzmann1()
#     result[i] = Boltzmann()


# plt.hist(np.around(result, 1), bins=210, edgecolor='black')

# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Data')

# # Show the plot
# plt.show()

# a = np.array(np.arange(0.00, 2.10, 0.01))
# print(a)
# print(np.around(a,1))

# create a csv file and write data into it
# Define the header
# header = ['Episode', 'Reward', 'Steps']

# # Specify the file name
# file_name = 'rl_episodes_data.csv'

# # Create and open the CSV file in write mode
# with open(file_name, mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write the header to the CSV file
#     writer.writerow(header)

# # close the writer
# file.close()
# # Inside your RL loop
# for episode in range(5):
#     # ... Run your RL algorithm ...

#     # After each episode, record relevant data
#     episode_data = [episode, episode+1, episode+2]

#     # Append the data to the CSV file
#     with open(file_name, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(episode_data)
#     file.close()

# def generate_output(N):
#     distb_level = np.arange(0.0, 2.1, 0.1)
#     total_numbers = 21
#     interval = N // total_numbers
#     remainder = N % total_numbers

#     for i in range(N-remainder):
#         integer = i // interval
#         print(f"{distb_level[integer]:.1f}")
#         # if i % interval == 0:
#         #     print(f'{current_number / 10:.1f}')
#         #     current_number += 1

# N = 60  # 这里可以替换成你想要的N值
# generate_output(N)

# import csv

# # Assuming you have three lists: ep_lengths, returns, and distbs

# ep_lengths = [10, 15, 20, 25]
# returns = [100, 150, 200, 250]
# distbs = [0.5, 0.6, 0.7, 0.8]

# # Combine the lists into a list of tuples for easier writing to CSV
# data = list(zip(ep_lengths, returns, distbs))

# # Specify the file name and open the CSV file in write mode
# file_name = 'episode_data.csv'

# with open(file_name, 'w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write the header
#     writer.writerow(['Episode Length', 'Returns', 'Distbs'])
    
#     # Write the data
#     writer.writerows(data)

# print(f'Data has been written to {file_name}')



# folder_path = "train_results_phoenix/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_18_11_29/seed_09209/torch_save"
# files = os.listdir(folder_path)
# model_files = [file for file in files if file.startswith("model") and file.endswith(".pt")]
# print(model_files)
# largest_suffix = 0
# selected_model = ""

# # choose the maximum suffix number in the model_files


# suffixes = [int(f.split('model')[1].split('.pt')[0]) for f in model_files if f != 'model.pt']
# max_suffix = max(suffixes)

# model_path = os.path.join(folder_path, f'model{max_suffix}.pt')

# print(model_path)


# Test reward functions
rpy = np.zeros(3, dtype=np.float64)
print(rpy)

