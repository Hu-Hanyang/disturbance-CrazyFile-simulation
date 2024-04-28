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
from phoenix_drone_simulation.envs.hover import DroneHoverBaseEnv
from phoenix_drone_simulation.envs.agents import CrazyFlieSimpleAgent, CrazyFlieBulletAgent, CrazyFlieBulletAgentWithAdversary
from phoenix_drone_simulation.envs.utils import deg2rad, rad2deg, get_assets_path, Boltzmann
from phoenix_drone_simulation.adversarial_generation.FasTrack_data.distur_gener import distur_gener, quat2euler


class DroneHoverEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 observation_noise=1,  # must be positive in order to add noise
                 domain_randomization: float = -1,
                 enable_reset_distribution=True,  # Hanyang: enable randomized intial states
                 distb_level=1.0,  # Hanyang: try different values to see the upperbound
                 adv_policy = None,
                 **kwargs):
        super(DroneHoverEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            enable_reset_distribution=enable_reset_distribution,
            **kwargs
        )


       # 
        self.disturbance_level = distb_level # 2.0  # 1.5 # Hanyang: try different values to see the upperbound
        self.id = 'DroneHoverFixedDistbEnv'
        self.adv_policy = adv_policy


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
        states = np.concatenate((angles, angular_rates), axis=0)

        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            if self.adv_policy is None:
                distb = np.zeros_like(action)
            else:
                distb,_,_ = self.adv_policy(states)
            self.physics.step_forward(action, distb)

            # Note: do not delete the following line due to >100 Hz sensor noise
            self.compute_observation()
            self.iteration += 1

        # add observation and action to history..
        next_obs = self.compute_history()
        # print(f"The next observation shape is: {next_obs.shape}")

        r = self.compute_reward(action)
        info = self.compute_info()
        done = self.compute_done()
        self.last_action = action
        return next_obs, r, done, info
    




class DroneAdvEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 observation_noise=1,  # must be positive in order to add noise
                 domain_randomization: float = -1,
                 enable_reset_distribution=True,  # Hanyang: enable randomized intial states
                 distb_level=1.0,  # Hanyang: try different values to see the upperbound
                 env_policy = None,
                 **kwargs):
        super(DroneAdvEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            enable_reset_distribution=enable_reset_distribution,
            **kwargs
        )


       # 
        self.disturbance_level = distb_level # 2.0  # 1.5 # Hanyang: try different values to see the upperbound
        self.id = 'DroneHoverFixedDistbEnv'
        self.env_policy = env_policy


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
        states = np.concatenate((angles, angular_rates), axis=0)
        # print("The disturbance shape is: ", len(dstb))
        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            if self.env_policy is None:
                distb = np.zeros_like(action)
            else:
                distb, _, _ = self.env_policy(states)
            self.physics.step_forward(action, distb)

            # Note: do not delete the following line due to >100 Hz sensor noise
            self.compute_observation()
            self.iteration += 1

        # add observation and action to history..
        next_obs = self.compute_history()
        # print(f"The next observation shape is: {next_obs.shape}")

        r = -self.compute_reward(action)
        info = self.compute_info()
        done = self.compute_done()
        self.last_action = action
        return next_obs, r, done, info