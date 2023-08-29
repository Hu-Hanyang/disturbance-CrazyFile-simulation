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
from adversarial_generation.FasTrack_data.distur_gener import * 



class PybulletPhysicsWithAdversary(PyBulletPhysics):
    def set_parameters(self, *args, **kwargs):
        super(PybulletPhysicsWithAdversary, self).set_parameters(*args, **kwargs)
        # Update PyBullet Physics
        self.bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1
        )

    def step_forward(self, action, dstb, *args, **kwargs):
        """ PyBullet physics with adversary effect included implementation.

        Parameters
        ----------
        action
        """
        # calculate current motor forces (incorporates delays with motor speeds)
        motor_forces, z_torque = self.drone.apply_action(action)

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.drone.apply_motor_forces(motor_forces)
        self.drone.apply_z_torque(z_torque)

        # === XL: add adversary effect
        self.drone.apply_x_torque(dstb[0])
        self.drone.apply_y_torque(dstb[1])

        # === add drag effect
        quat = self.drone.quaternion
        vel = self.drone.xyz_dot
        base_rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Simple draft model applied to the base/center of mass
        rpm = self.drone.x**2 * 25000
        drag_factors = -1 * self.drone.DRAG_COEFF * np.sum(2*np.pi*rpm/60)
        drag = np.dot(base_rot, drag_factors*np.array(vel))
        # print(f'Drag: {drag}')
        self.drone.apply_force(force=drag)

        # === Ground Effect
        apply_ground_eff, ge_forces = self.calculate_ground_effect(motor_forces)
        if apply_ground_eff and self.use_ground_effect:
            self.drone.apply_motor_forces(forces=ge_forces)

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()


class CrazyFlieBulletAgentWithAdversary(CrazyFlieAgent):
    def __init__(
            self,
            bc: bullet_client.BulletClient,
            control_mode: str,
            time_step: float,
            aggregate_phy_steps: int,
            **kwargs
    ):
        super().__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            bc=bc,
            control_mode=control_mode,
            file_name='cf21x_bullet.urdf',
            time_step=time_step,
            use_latency=True,
            use_motor_dynamics=True,  # use first-order motor dynamics
            **kwargs
        )
    
    def apply_x_torque(self, torque):
        """Apply torque responsible for roll."""
        self.bc.applyExternalTorque(
            self.body_unique_id,
            4,  # center of mass link
            torqueObj=[torque, 0, 0],
            flags=pb.LINK_FRAME
        )

    def apply_y_torque(self, torque):
        """Apply torque responsible for pitch."""
        self.bc.applyExternalTorque(
            self.body_unique_id,
            4,  # center of mass link
            torqueObj=[0, torque, 0],
            flags=pb.LINK_FRAME
        )


class DroneHoverBulletEnvWithAdversary(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithAdversary, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        # - torques range of x,y,z axes (2*Umax) as customized adversary bound
        # - a disturbance generator (lambda func)
        # self.dstb_space = gym.spaces.Box(low=np.array([-2*5.3*10**-3, -2*5.3*10**-3, -2*1.43*10**-4]), 
        #                                 high=np.array([2*5.3*10**-3,  2*5.3*10**-3,  2*1.43*10**-4]), 
        #                                 dtype=np.float32)
        # self.dstb_space = gym.spaces.Box(low=np.array([-5.3*10**-3, -5.3*10**-3, -1.43*10**-4]), 
        #                                 high=np.array([5.3*10**-3,  5.3*10**-3,  1.43*10**-4]), 
        #                                 dtype=np.float32)
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        # self.dstb_gen   = lambda x: self.dstb_space.sample() 
        self.disturbance_level = 1.5 # Dmax = uMax * disturbance_level
        # self.dstb_gen = lambda x: np.array([0,0,0])


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverBulletEnvWithAdversary, self)._setup_task_specifics()

        # === Reset camera position
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.8,
            cameraYaw=10,
            cameraPitch=-50
        )

    """
    XL: Rewrite this method from parent class (DroneBaseEnv), to include adversary agent and physics
    """
    def _setup_simulation(
            self,
            physics: str,
    ) -> None:
        r"""Create world layout, spawn agent and obstacles.

        Takes the passed parameters from the class instantiation: __init__().
        """
        # reset some variables that might be changed by DR -- this avoids errors 
        # when calling the render() method after training.
        self.g = self.G
        self.time_step = self.TIME_STEP

        # also add PyBullet's data path
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = self.bc.loadURDF("plane.urdf")
        # Load 10x10 Walls
        pb.loadURDF(os.path.join(get_assets_path(), "room_10x10.urdf"), useFixedBase=True)
        # random spawns

        if self.drone_model == 'cf21x_bullet':
            self.drone = CrazyFlieBulletAgent(bc=self.bc, **self.agent_params)
        elif self.drone_model == 'cf21x_sys_eq':
            self.drone = CrazyFlieSimpleAgent(bc=self.bc, **self.agent_params)
        elif self.drone_model == 'cf21x_bullet_adversary':
            print("Loading drone model:".format(self.drone_model))
            self.drone = CrazyFlieBulletAgentWithAdversary(bc=self.bc, **self.agent_params)
        else:
            raise NotImplementedError

        # Setup forward dynamics - Instantiates a particular physics class.
        setattr(phoenix_physics, 'PybulletPhysicsWithAdversary', PybulletPhysicsWithAdversary) # XL: add adversary physics to the module
        assert hasattr(phoenix_physics, physics), f'Physics={physics} not found.'
        physics_cls = getattr(phoenix_physics, physics)  # get class reference
        
        if physics == "PybulletPhysicsWithAdversary":  # XL: assert the drone and its associated physics
            assert self.drone_model == "cf21x_bullet_adversary"

        # call class constructor
        self.physics = physics_cls(
            self.drone,
            self.bc,
            time_step=self.time_step,  # 1 / sim_frequency
        )

        # Setup task specifics
        self._setup_task_specifics()

    """
    XL: Rewrite this method from parent class (DroneBaseEnv), to include use of disturbance
    """
    def step(
            self,
            action: np.ndarray,
    ) -> tuple:
        """Step the simulation's dynamics once forward.

        This method follows the interface of the OpenAI Gym.

        Parameters
        ----------
        action: array
            Holding the control commands for the agent. 

        Returns
        -------
        observation (object)
            Agent's observation of the current environment
        reward (float)
            Amount of reward returned after previous action
        done (bool)
            Whether the episode has ended, handled by the time wrapper
        info (dict)
            contains auxiliary diagnostic information such as the cost signal
        """

        # XL: This is special in our adversary Env for generating 
        # disturbance from HJ reachability
        angles = quat2euler(self.drone.get_state()[3:7])
        angular_rates = self.drone.get_state()[10:13]
        # dstb = self.dstb_gen(x) # generate disturbance
        # Disturbance gives a six dimensional vector, includes [roll_angle, pitch_angle, yaw_angle, roll_rate, pitch_rate, yaw_rate]
        # yaw_angle and yaw_rate are not used in the disturbance model
        # combine angles and angular rates together to form a [6,1] list
        states = np.concatenate((angles, angular_rates), axis=0)
        _, dstb = distur_gener(states, self.disturbance_level) 
        # Hanyang: try to do not add distb 
        # dstb = (0.0, 0.0, 0.0)

        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            self.physics.step_forward(action, dstb)

            # Note: do not delete the following line due to >100 Hz sensor noise
            self.compute_observation()
            self.iteration += 1

        # add observation and action to history..
        next_obs = self.compute_history()

        r = self.compute_reward(action)
        info = self.compute_info()
        done = self.compute_done()
        self.last_action = action
        return next_obs, r, done, info
    
    """
    XL: Rewrite this method from parent class (DroneBaseEnv), to enable proper rendering
    """
    def render(self, mode="human"):
        super(DroneHoverBulletEnvWithAdversary, self).render(mode)

    """
    XL: Rewrite this method from parent class (DroneBaseEnv), to change the criteria of done
    """
    def compute_done(self) -> bool:
        """ Note: the class is wrapped by Gym's Time-wrapper, which returns
        done=True when T >= time_limit."""
        rp = self.drone.rpy[:2]  # [rad]
        d = deg2rad(75) # by default 60 deg
        z_limit = self.drone.xyz[2] < 0.2
        rpy_limit = rp[np.abs(rp) > d].any()

        rpy_dot = self.drone.rpy_dot  # in rad/s
        rpy_dot_limit = rpy_dot[rad2deg(np.abs(rpy_dot)) > 1000].any() # by default 300 deg/s, increase it to handle more adversary effect

        done = True if rpy_limit or rpy_dot_limit or z_limit else False
        return done

    


def test_env(env_id):
    if "Adversary" in env_id:
        assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
        register(id=env_id, entry_point="{}:{}".format(
            DroneHoverBulletEnvWithAdversary.__module__, 
            DroneHoverBulletEnvWithAdversary.__name__), 
            max_episode_steps=500,)

    now = datetime.now()
    tim = "{}_{}_{}_{}_{}_{}".format(now.strftime("%Y"),now.strftime("%m"),now.strftime("%d"),
                                    now.strftime("%H"),now.strftime("%M"), now.strftime("%S"))
    env = gym.make(env_id)

    while True:
        done = False
        env.render()  # make GUI of PyBullet appear
        x = env.reset()
        while not done:
            random_action = env.action_space.sample()
            x, reward, done, info = env.step(action=random_action)
            time.sleep(0.05)  # FPS: 20 (real-time)



def start_training(algo, env_id):
    env_id = env_id
    # if "Adversary" in env_id:
    #     assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    #     register(id=env_id, entry_point="{}:{}".format(
    #         DroneHoverBulletEnvWithAdversary.__module__, 
    #         DroneHoverBulletEnvWithAdversary.__name__), 
    #         max_episode_steps=500,)



    # Create a seed for the random number generator
    random_seed = int(time.time()) % 2 ** 16

    # I usually save my results into the following directory:
    default_log_dir = f"./runs/original_ppo/our_distb"

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
    env = gym.make(env_id)
    # Important note: PyBullet necessitates to call env.render()
    # before env.reset() to display the GUI!
    env.render() 
    while True:
        obs = env.reset()
        done = False
        while not done:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            action, value, *_ = model.actor_critic(obs)
            obs, reward, done, info = env.step(action)

            time.sleep(0.05)
            if done:
                obs = env.reset()


    



if __name__ == "__main__":
    # == setting attr class to module
    # print(getattr(phoenix_physics, 'SimplePhysics'))
    # setattr(phoenix_physics, 'PybulletPhysicsWithAdversary', PybulletPhysicsWithAdversary)
    # print(phoenix_physics)
    # assert hasattr(phoenix_physics, 'PybulletPhysicsWithAdversary')

    # == test customized env loop
    # test_env(env_id='DroneHoverBulletEnvWithAdversary-v0')

    # == start training with ppo
    start_training(algo="ppo", env_id="DroneHoverBulletEnvWithAdversary-v0")
