import os
import gym
import time
import numpy as np
import pybullet as pb
import pybullet_data
import phoenix_drone_simulation.envs.physics as phoenix_physics
from PIL import Image
from phoenix_drone_simulation.envs.physics import PybulletPhysicsWithAdversary
from phoenix_drone_simulation.envs.base import DroneBaseEnv
from phoenix_drone_simulation.envs.agents import CrazyFlieSimpleAgent, CrazyFlieBulletAgent, CrazyFlieBulletAgentWithAdversary
from phoenix_drone_simulation.envs.utils import deg2rad, rad2deg, get_assets_path
from phoenix_drone_simulation.adversarial_generation.FasTrack_data.distur_gener import distur_gener, quat2euler