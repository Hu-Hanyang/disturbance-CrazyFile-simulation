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
from phoenix_drone_simulation.envs.utils import deg2rad, rad2deg, get_assets_path, Boltzmann
from phoenix_drone_simulation.adversarial_generation.FasTrack_data.distur_gener import distur_gener, quat2euler


class DroneHoverBaseEnv(DroneBaseEnv):
    def __init__(
            self,
            physics,
            control_mode: str,
            drone_model: str,
            observation_noise=1,  # must be positive in order to add noise
            domain_randomization: float = 0.10,  # use 10% DR as default
            target_pos: np.ndarray = np.array([0, 0, 1.0], dtype=np.float32),
            target_xyz_dot: np.ndarray = np.array([0, 0, 0], dtype=np.float32),  # Hanyang: target velocity
            target_rpy: np.ndarray = np.array([0, 0, 0], dtype=np.float32),  # Hanyang: target orientation
            target_rpy_dot: np.ndarray = np.array([0, 0, 0], dtype=np.float32),  # Hanyang: target angular velocity
            sim_freq=200,  # in Hz
            aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
            observation_frequency=100,  # in Hz
            penalty_action_rate: float = 1e-3,  # Hanyang: penalty for action changing rate
            penalty_angle: float = 1e-2,  # Hanyang: penalty for rpy rate (roll rate, pitch rate, yaw rate)
            penality_angle_rate: float = 1e-3,  # Hanyang: penalty for rpy rate (roll rate, pitch rate, yaw rate)
            penalty_spin: float = 0.0,  # Hanyang: penalty for yaw rate
            penalty_terminal: float = 1.0,  # Hanyang: try larger crash penalty,original is 100
            penalty_velocity: float = 1e-3,  # Hanyang: penalty for velocity (dot(x, y, z))
            penalty_z: float = 0.0,  # Hanyang: penalty for the z position
            **kwargs
    ):
        # === Hover task specific attributes
        # must be defined before calling super class constructor:
        self.target_pos = target_pos  # used in _computePotential()
        self.target_xyz_dot = target_xyz_dot
        self.target_rpy = target_rpy
        self.target_rpy_dot = target_rpy_dot

        # Hanyang: penalty parameters and penalty logs
        self.ARP = penalty_action_rate
        self.penalty_angle = penalty_angle
        self.penalty_angle_rate = penality_angle_rate
        self.penalty_spin = penalty_spin
        self.penalty_terminal = penalty_terminal
        self.penalty_velocity = penalty_velocity
        self.penalty_z = penalty_z
        self.penalty_crash = 0.0
        self.penalty = 0.0
        self.penalty_rpy = 0.0
        self.penalty_rpy_dot = 0.0

        self.penalty_log = 0.0
        self.penalty_rpy_log = 0.0
        self.penalty_crash_log = 0.0
        self.penalty_z_log = 0.0
        self.penalty_rpy_dot_log = 0.0
        self.penalty_velocity_log = 0.0

        # === Costs: The following constants are used for cost calculation:
        self.vel_limit = 0.25  # [m/s]
        self.roll_pitch_limit = deg2rad(10)  # [rad]
        self.rpy_dot_limit = deg2rad(200)  # [rad/s]
        self.x_lim = 0.10
        self.y_lim = 0.10
        self.z_lim = 1.20

        # task specific parameters - init drone state
        init_xyz = np.array([0, 0, 1], dtype=np.float32)
        init_rpy = np.zeros(3)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        # Hanyang: disturbance level
        self.disturbance_level = 0.0

        super(DroneHoverBaseEnv, self).__init__(
            control_mode=control_mode,
            drone_model=drone_model,
            init_xyz=init_xyz,
            init_rpy=init_rpy,
            init_xyz_dot=init_xyz_dot,
            init_rpy_dot=init_rpy_dot,
            physics=physics,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            sim_freq=sim_freq,
            aggregate_phy_steps=aggregate_phy_steps,
            observation_frequency=observation_frequency,
            **kwargs
        )

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.95, 0.1, 0.05, 0.4],
        )
        # Spawn visual without collision shape
        self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )

        # === Set camera position
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-70
        )

    def compute_done(self) -> bool:
        """ Note: the class is wrapped by Gym's Time-wrapper, which returns
        done=True when T >= time_limit."""
        rp = self.drone.rpy[:2]  # [rad]
        d = deg2rad(60)
        z_limit = self.drone.xyz[2] < 0.2
        rpy_limit = rp[np.abs(rp) > d].any()

        rpy_dot = self.drone.rpy_dot  # in rad/s
        rpy_dot_limit = rpy_dot[rad2deg(np.abs(rpy_dot)) > 300].any()

        done = True if rpy_limit or rpy_dot_limit or z_limit else False
        return done

    def compute_info(self) -> dict:
        state = self.drone.get_state()
        c = 0.
        info = {}
        # xyz bounds
        x, y, z = state[:3]
        if np.abs(x) > self.x_lim or np.abs(y) > self.y_lim or z > self.z_lim:
            c = 1.
            info['xyz_limit'] = state[:3]
        # roll pitch bounds
        rpy = self.drone.rpy
        if (np.abs(rpy[:2]) > self.roll_pitch_limit).any():
            c = 1.
            info['rpy'] = rpy
        # linear velocities
        if (np.abs(state[10:13]) > self.vel_limit).any():
            c = 1.
            info['xzy_dot'] = state[10:13]
        # angular velocities
        if (np.abs(state[13:16]) > self.rpy_dot_limit).any():
            c = 1.
            info['rpy_dot'] = state[13:16] * 180 / np.pi
        # update ron visuals when costs are received
        # self.violates_constraints(True if c > 0 else False)
        info['cost'] = c

        # Hanyang: add disturbance level to info
        info['disturbance_level'] = self.disturbance_level
        return info

    def compute_observation(self) -> np.ndarray:

        if self.observation_noise > 0:  # add noise only for positive values
            if self.iteration % self.obs_rate == 0:
                # === 100 Hz Part ===
                # update state information with 100 Hz (except for rpy_dot)
                # apply noise to perfect simulation state:
                xyz, vel, rpy, omega, acc = self.sensor_noise.add_noise(
                    pos=self.drone.xyz,
                    vel=self.drone.xyz_dot,
                    rot=self.drone.rpy,
                    omega=self.drone.rpy_dot,
                    acc=np.zeros(3),  # irrelevant
                    dt=1/self.SIM_FREQ
                )
                quat = np.asarray(self.bc.getQuaternionFromEuler(rpy))
                self.state = np.concatenate(
                    [xyz, quat, vel, omega, self.drone.last_action])
            else:
                # print("The iteration is: ", self.iteration)
                # === 200 Hz Part ===
                # This part is run with 200Hz, re-use Kalman Filter values:
                xyz, quat, vel = self.state[0:3], self.state[3:7], self.state[7:10]
                # read Gyro data with 500 Hz and add noise:
                omega = self.sensor_noise.add_noise_to_omega(
                    omega=self.drone.rpy_dot, dt=1/self.SIM_FREQ)

            # apply low-pass filtering to gyro (happens with 100Hz):
            omega = self.gyro_lpf.apply(omega)
            obs = np.concatenate([xyz, quat, vel, omega])  # Hanyang TODO: the dim of the obs is 13 which is not consistent with the result
        else:
            # no observation noise is applied
            obs = self.drone.get_state()
        return obs

    def compute_potential(self) -> float:
        """Euclidean distance from current ron position to target position."""
        return np.linalg.norm(self.drone.xyz - self.target_pos)

    def compute_reward(self, action) -> float:  # Hanyang: redesign, 2.2.2024
        # Determine penalties
        act_diff = action - self.drone.last_action
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)

        # penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)
        penalty_action_rate = self.ARP * np.linalg.norm(act_diff)
        penalty_rpy = self.penalty_angle * np.linalg.norm(self.drone.rpy)
        penalty_rpy_dot = self.penalty_angle_rate * np.linalg.norm(self.drone.rpy_dot)
        penalty_spin = self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        penalty_terminal = self.penalty_terminal if self.compute_done() else 0.  # Hanyang: try larger crash penalty
        penalty_velocity = self.penalty_velocity * np.linalg.norm(
            self.drone.xyz_dot)
        
        # Hanyang: split the reward function to 2 stages
        # Stage 1: penalize the large rpy and the action rate
        threshold_rpy = 15*np.pi/180  # 15 degree
        if np.any(np.abs(self.drone.rpy) > threshold_rpy):
            penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_terminal])
        else:
            penalties = np.sum([-penalty_rpy, -penalty_velocity, -penalty_rpy_dot, penalty_terminal])
            
        # # Hanyang: the current valid rewards are: penalty_rpy, penalty_spin, penalty_velocity and penalty_terminal
        # penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_spin,
        #                     penalty_velocity, penalty_action_rate, penalty_terminal])
        
        # L2 norm:
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        reward = -dist - penalties
        return reward

    def task_specific_reset(self):
        """Initialize task states. Called by reset()."""
        # set random offset for position
        # Note: use copy() to avoid chaning original initial values
        pos = self.init_xyz.copy()
        xyz_dot = self.init_xyz_dot.copy()
        rpy_dot = self.init_rpy_dot.copy()
        quat = self.init_quaternion.copy()

        if self.enable_reset_distribution:  # Hanyang: default True
            pos_lim = 0.25   # should be at least 0.15 for hover task since position PID shoots up to (0,0,1.15)

            pos += np.random.uniform(-pos_lim, pos_lim, size=3)

            init_angle = np.pi/6
            rpy = np.random.uniform(-init_angle, init_angle, size=3)
            yaw_init = 2 * np.pi
            rpy[2] = np.random.uniform(-yaw_init, yaw_init)
            quat = self.bc.getQuaternionFromEuler(rpy)

            # set random initial velocities
            vel_lim = 0.1  # default: 0.05
            xyz_dot = xyz_dot + np.random.uniform(-vel_lim, vel_lim, size=3)

            rpy_dot_lim = deg2rad(200)  # default: 10
            rpy_dot = rpy_dot + np.random.uniform(-rpy_dot_lim, rpy_dot_lim, size=3)
            rpy_dot[2] = np.random.uniform(-deg2rad(20), deg2rad(20))

            # init low pass filter(s) with new values:
            self.gyro_lpf.set(x=rpy_dot)

            # set drone internals
            self.drone.x = np.random.normal(self.drone.HOVER_X, scale=0.02,
                                            size=(4,))
            self.drone.y = self.drone.K * self.drone.x
            self.drone.action_buffer = np.clip(
                np.random.normal(self.drone.HOVER_ACTION, 0.02,
                                 size=self.drone.action_buffer.shape), -1, 1)
            self.drone.last_action = self.drone.action_buffer[-1, :]


        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )
        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ rpy_dot
        )

""" ==================
        PWM control
    ==================
"""


class DroneHoverSimpleEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=1,
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverSimpleEnv, self).__init__(
            control_mode=control_mode,
            drone_model='cf21x_sys_eq',
            physics='SimplePhysics',
            # use 100 Hz since no motor dynamics and PID is used
            sim_freq=100,
            aggregate_phy_steps=aggregate_phy_steps,
            **kwargs
        )


class DroneHoverBulletEnv(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
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
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        # self.dstb_gen   = lambda x: self.dstb_space.sample() 
        self.disturbance_level = 0.0 # 2.0  # 1.5 # Hanyang: try different values to see the upperbound
        self.id = 'DroneHoverBulletEnvWithAdversary'


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


class DroneHoverBulletEnvWithRandomHJAdversary(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithRandomHJAdversary, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        
        # Hanyang: generate random number in the range of [0.0, 2.0] with 1 decimal place and use it as the distburance level
        # self.disturbance_level = round(np.random.uniform(0.0, 2.1), 1)
        # self.disturbance_level = 0.0
        self.disturbance_level = Boltzmann()

        self.id = 'DroneHoverBulletEnvWithRandomHJAdversary'

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
        super(DroneHoverBulletEnvWithRandomHJAdversary, self)._setup_task_specifics()

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
        # Hanyang: add Boltzmann distribution to generate random disturbance level
        # self.disturbance_level = Boltzmann()
        # self.disturbance_level = round(np.random.uniform(0.0, 2.1), 1)

        _, dstb = distur_gener(states, self.disturbance_level) 

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


class DroneHoverBulletEnvWithoutAdversary(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithoutAdversary, self).__init__(
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
        self.id = 'DroneHoverBulletEnvWithoutAdversary'


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverBulletEnvWithoutAdversary, self)._setup_task_specifics()

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
        dstb = (0.0, 0.0, 0.0)

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
        super(DroneHoverBulletEnvWithoutAdversary, self).render(mode)

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

class DroneHoverBulletEnvWithRandomAdversary(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithRandomAdversary, self).__init__(
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
        self.id = 'DroneHoverBulletEnvWithRandomAdversary'


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverBulletEnvWithRandomAdversary, self)._setup_task_specifics()

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
        super(DroneHoverBulletEnvWithRandomAdversary, self).render(mode)

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

class DroneHoverBulletEnvWithAdversaryInitial(DroneHoverBaseEnv):
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithAdversaryInitial, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        # self.dstb_gen   = lambda x: self.dstb_space.sample() 
        self.disturbance_level = 1.5  # 2.0  # 1.5 # Hanyang: try different values to see the upperbound
        self.id = 'DroneHoverBulletEnvWithAdversaryInitial'

        # change the varibale here to save the config in env_config.jason
        self.initial_angle = np.pi/4
        self.vel_lim = 0.1
        self.rpy_dot_limit = deg2rad(300)


    """
    XL: Rewrite this method from parent class
    """
    def _setup_task_specifics(self):
        super(DroneHoverBulletEnvWithAdversaryInitial, self)._setup_task_specifics()

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
    
    """
    Hanyang: try different range of initial states
    """
    def task_specific_reset(self):
        # set random offset for position
        # Note: use copy() to avoid chaning original initial values
        # print("Let's try different range of initial states:")
        pos = self.init_xyz.copy()
        xyz_dot = self.init_xyz_dot.copy()
        rpy_dot = self.init_rpy_dot.copy()
        quat = self.init_quaternion.copy()

        if self.enable_reset_distribution:  # Hanyang: default True
            pos_lim = 0.25   # should be at least 0.15 for hover task since position PID shoots up to (0,0,1.15)

            pos += np.random.uniform(-pos_lim, pos_lim, size=3)

            init_angle = self.initial_angle  # original: pi/6
            rpy = np.random.uniform(-init_angle, init_angle, size=3)
            yaw_init = 2 * np.pi
            rpy[2] = np.random.uniform(-yaw_init, yaw_init)
            quat = self.bc.getQuaternionFromEuler(rpy)

            # set random initial velocities
            vel_lim = self.vel_lim  # original: 0.1
            xyz_dot = xyz_dot + np.random.uniform(-vel_lim, vel_lim, size=3)

            rpy_dot_lim = self.rpy_dot_limit  # original: 200
            rpy_dot = rpy_dot + np.random.uniform(-rpy_dot_lim, rpy_dot_lim, size=3)
            rpy_dot[2] = np.random.uniform(-deg2rad(20), deg2rad(20))

            # init low pass filter(s) with new values:
            self.gyro_lpf.set(x=rpy_dot)

            # set drone internals
            self.drone.x = np.random.normal(self.drone.HOVER_X, scale=0.02, size=(4,))
            self.drone.y = self.drone.K * self.drone.x
            self.drone.action_buffer = np.clip(
                np.random.normal(self.drone.HOVER_ACTION, 0.02,
                                 size=self.drone.action_buffer.shape), -1, 1)
            self.drone.last_action = self.drone.action_buffer[-1, :]

        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )
        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ rpy_dot
        )

    
class DroneHoverBulletEnvWithCurriculumHJAdversary(DroneHoverBaseEnv):  #TODO: Not finished yet
    def __init__(self,
                 aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
                 control_mode='PWM',
                 **kwargs):
        super(DroneHoverBulletEnvWithCurriculumHJAdversary, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet_adversary',  # CrazyFlieBulletAgentWithAdversary
            physics='PybulletPhysicsWithAdversary',  # physics env, not the concept in rl
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )


        # XL: set properties of input disturbances
        self.dstb_space = gym.spaces.Box(low=np.array([-1*10**-3, -1*10**-3, -1*10**-4]), 
                                        high=np.array([1*10**-3,  1*10**-3,  1*10**-4]), 
                                        dtype=np.float32)
        
        # Hanyang: generate random number in the range of [0.0, 2.0] with 1 decimal place and use it as the distburance level
        # self.disturbance_level = round(np.random.uniform(0.0, 2.1), 1)
        # self.disturbance_level = 0.0
        self.epoch = 0
        self.disturbance_level = Boltzmann()

        self.id = 'DroneHoverBulletEnvWithRandomHJAdversary'

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
        super(DroneHoverBulletEnvWithCurriculumHJAdversary, self)._setup_task_specifics()

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
        # Hanyang: add Boltzmann distribution to generate random disturbance level
        # self.disturbance_level = Boltzmann()
        # self.disturbance_level = round(np.random.uniform(0.0, 2.1), 1)

        _, dstb = distur_gener(states, self.disturbance_level) 

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

