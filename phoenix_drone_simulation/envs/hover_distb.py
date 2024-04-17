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


class DroneHoverFixedDistbEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 observation_noise=1,  # must be positive in order to add noise
                 domain_randomization: float = -1,
                 enable_reset_distribution=True,
                 distb_level=1.0,  # Hanyang: try different values to see the upperbound
                 **kwargs):
        super(DroneHoverFixedDistbEnv, self).__init__(
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
        _, distb = distur_gener(states, self.disturbance_level) 
        # print("The disturbance shape is: ", len(dstb))
        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
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
    


class DroneHoverBoltzmannDistbEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 observation_noise=1,  # must be positive in order to add noise
                 domain_randomization: float = -1,  # use 10% DR as default
                 enable_reset_distribution=True, 
                 **kwargs):
        super(DroneHoverBoltzmannDistbEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            enable_reset_distribution=enable_reset_distribution,
            **kwargs
        )


        #TODO: Hanyang: change the Boltzmann temperature hyperparameters here
        self.disturbance_level = Boltzmann()

        self.id = 'DroneHoverBoltzmannDistbEnv'

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        This function is called after agent encountered terminal state.

        Returns
        -------
        array
            holding the observation of the initial state
        """
        self.iteration = 0
        # disable rendering before resetting
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        if self.stored_state_id >= 0:
            self.bc.restoreState(self.stored_state_id)
        else:
            # Restoring a saved state circumvents the necessity to load all
            # bodies again..
            self.stored_state_id = self.bc.saveState()
        self.drone.reset()  # resets only internals such as x, y, last action
        self.task_specific_reset()
        self.apply_domain_randomization()

        # init low pass filter(s) with new values:
        self.gyro_lpf.set(x=self.drone.rpy_dot)

        # collect information from PyBullet simulation
        """Gather information from PyBullet about drone's current state."""
        self.drone.update_information()
        self.old_potential = self.compute_potential()
        self.state = self.drone.get_state()
        if self.use_graphics:  # enable rendering again after resetting
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
        obs = self.compute_observation()

        # fill history buffers
        # add observation and action to history..
        N = self.observation_history.maxlen
        [self.observation_history.append(obs) for _ in range(N)]
        action = self.drone.last_action
        [self.action_history.append(action) for _ in range(N)]
        self.last_action = action
        obs = self.compute_history()

        # Hanyang: set the disturbance level to a random number in the range of [0.0, 2.0] with 1 decimal place
        self.disturbance_level = Boltzmann()

        return obs

    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverBoltzmannDistbEnv, self)._setup_task_specifics()

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
        states = np.concatenate((angles, angular_rates), axis=0)

        _, distb = distur_gener(states, self.disturbance_level) 

        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            self.physics.step_forward(action, distb)

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
    


class DroneHoverNoDistbEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverNoDistbEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        # self.dstb_gen   = lambda x: self.dstb_space.sample() 
        self.disturbance_level = 0.0 # Dmax = uMax * disturbance_level
        self.id = 'DroneHoverNoDistbEnv'


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverNoDistbEnv, self)._setup_task_specifics()

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
        # _, dstb = distur_gener(states, self.disturbance_level) 
        # Hanyang: no disturbances
        distb = (0.0, 0.0, 0.0)

        for _ in range(self.aggregate_phy_steps):
            # Note:
            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            self.physics.step_forward(action, distb)

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
        super(DroneHoverNoDistbEnv, self).render(mode)

    # """
    # XL: Rewrite this method from parent class (DroneBaseEnv), to change the criteria of done
    # """
    # def compute_done(self) -> bool:
    #     """ Note: the class is wrapped by Gym's Time-wrapper, which returns
    #     done=True when T >= time_limit."""
    #     rp = self.drone.rpy[:2]  # [rad]
    #     d = deg2rad(75) # by default 60 deg
    #     z_limit = self.drone.xyz[2] < 0.2
    #     rpy_limit = rp[np.abs(rp) > d].any()

    #     rpy_dot = self.drone.rpy_dot  # in rad/s
    #     rpy_dot_limit = rpy_dot[rad2deg(np.abs(rpy_dot)) > 1000].any() # by default 300 deg/s, increase it to handle more adversary effect

    #     done = True if rpy_limit or rpy_dot_limit or z_limit else False
    #     return done

class DroneHoverRandomDistbEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverRandomDistbEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        self.dstb_gen   = lambda x: self.dstb_space.sample() 
        self.id = 'DroneHoverRandomDistbEnv'


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverRandomDistbEnv, self)._setup_task_specifics()

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
        # _, dstb = distur_gener(states, self.disturbance_level) 
        x = self.drone.get_state()
        dstb = self.dstb_gen(x)
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
        super(DroneHoverRandomDistbEnv, self).render(mode)

    # """
    # XL: Rewrite this method from parent class (DroneBaseEnv), to change the criteria of done
    # """
    # def compute_done(self) -> bool:
    #     """ Note: the class is wrapped by Gym's Time-wrapper, which returns
    #     done=True when T >= time_limit."""
    #     rp = self.drone.rpy[:2]  # [rad]
    #     d = deg2rad(75) # by default 60 deg
    #     z_limit = self.drone.xyz[2] < 0.2
    #     rpy_limit = rp[np.abs(rp) > d].any()

    #     rpy_dot = self.drone.rpy_dot  # in rad/s
    #     rpy_dot_limit = rpy_dot[rad2deg(np.abs(rpy_dot)) > 1000].any() # by default 300 deg/s, increase it to handle more adversary effect

    #     done = True if rpy_limit or rpy_dot_limit or z_limit else False
    #     return done
