"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Xubo
Date: 2022-11-23
Location: SFU Mars Lab
"""
import os, sys
import numpy as np
import abc
import pybullet as pb
from pybullet_utils import bullet_client
import pybullet_data
from typing import Tuple
import gym
import time
from datetime import datetime
from gym.envs.registration import register
import torch

import phoenix_drone_simulation.envs.physics as phoenix_physics
from phoenix_drone_simulation.envs.base import DroneBaseEnv
from phoenix_drone_simulation.envs.utils import deg2rad, rad2deg, get_assets_path
from phoenix_drone_simulation.envs.hover import DroneHoverBaseEnv
from phoenix_drone_simulation.envs.physics import PyBulletPhysics
from phoenix_drone_simulation.envs.agents import CrazyFlieAgent, CrazyFlieBulletAgent, CrazyFlieSimpleAgent
from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.adversarial_generation.FasTrack_data.distur_gener import distur_gener, quat2euler


# def test_env(env_id):
#     if "Adversary" in env_id:
#         assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
#         register(id=env_id, entry_point="{}:{}".format(
#             DroneHoverBulletEnvWithAdversary.__module__, 
#             DroneHoverBulletEnvWithAdversary.__name__), 
#             max_episode_steps=500,)

#     now = datetime.now()
#     tim = "{}_{}_{}_{}_{}_{}".format(now.strftime("%Y"),now.strftime("%m"),now.strftime("%d"),
#                                     now.strftime("%H"),now.strftime("%M"), now.strftime("%S"))
#     env = gym.make(env_id)

#     while True:
#         done = False
#         env.render()  # make GUI of PyBullet appear
#         x = env.reset()
#         while not done:
#             random_action = env.action_space.sample()
#             x, reward, done, info = env.step(action=random_action)
#             time.sleep(0.05)  # FPS: 20 (real-time)



def start_training(algo, env_id):
    env_id = env_id

    # Create a seed for the random number generator
    random_seed = int(time.time()) % 2 ** 16   # 40226 

    # I usually save my results into the following directory:
    default_log_dir = f"train_results_phoenix"

    # NEW: use algorithms implemented in phoenix_drone_simulation:
    # 1) Setup learning model
    model = Model(
        alg=algo,  # choose between: trpo, ppo
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )
    model.compile()

    start_time = time.perf_counter()

    # 2) Train model - it takes typically at least 100 epochs for training
    model.fit(epochs=300)

    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration}. \n")
    # 3) Benchmark the f
    # inal policy and save results into `returns.csv`
    model.eval()

    # 4) visualize trained PPO model
    # env = gym.make(env_id)
    # # Important note: PyBullet necessitates to call env.render()
    # # before env.reset() to display the GUI!
    # env.render() 
    # while True:
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         obs = torch.as_tensor(obs, dtype=torch.float32)
    #         action, value, *_ = model.actor_critic(obs)
    #         obs, reward, done, info = env.step(action)

    #         time.sleep(0.05)
    #         if done:
    #             obs = env.reset()


    



if __name__ == "__main__":
    # == test customized env loop
    # test_env(env_id='DroneHoverBulletEnvWithAdversary-v0')

    # == start training with ppo
    # start_training(algo="ppo", env_id="DroneHoverBulletEnvWithAdversary-v0")
    start_training(algo='ppo', env_id='DroneHoverBulletEnvWithoutAdversary-v0')  # no disturbance during training
