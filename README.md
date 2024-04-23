# Quadrotor Stabilization
This repository contains the implementation of the robust quadrotor stabilization research work.

## Installation
The simulation environment is based on the [OptimizedDP library](https://github.com/SFU-MARS/optimized_dp) and the [phoenix-drone-simulation](https://github.com/SvenGronauer/phoenix-drone-simulation.git).

0. Install the [pytorch](https://pytorch.org/).

1. First, run the following command to create a virtual environment named quadrotor and install the required packages (this process would take some time):
``conda env create -f environment.yml``

2. Then install the odp package (from the Optimized_DP library ) by:
```
cd adversarial_generation
pip install -e.
cd ..
```

3. One method to install the phoenix-drone-simulation package is to comment out the `mpy4pi` in the `install_requires` in the file `phoenix_drone_simulation/setup.py`, and use the following commands:
```
$ git clone https://github.com/SvenGronauer/phoenix-drone-simulation
$ cd phoenix-drone-simulation/
$ pip install -e .
```

The other method is to use `git submodule add` command. Unfortunately, I have not tried this before so I can not give any advice.

4. Finally install other dependencies:
```
pip install joblib
pip install tensorboard
conda install mpy4pi [reference](https://stackoverflow.com/questions/74427664/error-could-not-build-wheels-for-mpi4py-which-is-required-to-install-pyproject)
conda install pandas
```

Sorry for the complex installations, we will sort them up later.

2023.8.29 Update:
It seems there's something wrong with `pip install heterocl` or `pip install heterocl==0.1` and `pip install heterocl==0.3`.

## Information
### Agent 
class CrazyFlieAgent:
obs_dim = 22
act_dim = 4

# Working Logs
## Basic commands
### 1. Training: 
#### Method1 
`python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnvWithAdversary-v0`
Its default logdir is `/localhome/hha160/projects/disturbance-CrazyFile-simulation/train_results/`.
∆ TODO: This method takes too many paraller, needs to be changed.
Logs of method1:
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |  else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- |----------- |----------- |
| | | DroneHoverBulletEnvWithAdversary-v0 | ppo |  | None | Unknown | 1.5 | Nothing. |
#### Method2 `python adversaryhover_phoenix.py`
Its default logdir is `results_train_crazyflie`.
Logs of the env 'DroneHoverBulletEnvWithAdversary':
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- | ----------- |----------- |
| 2023_08_31_11_44 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_08_31_11_48/seed_40226 | None | Not bad | 1.5 | Nothing. |
| 2023_09_02_13_22 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_02_13_22/seed_40226 | None | Unknown | 2.0 | Nothing. |
| 2023_09_03_12_11 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_03_12_11/seed_40226 | None | Unknown | 1.0 | Nothing. |
| 2023_09_04_11_24 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_04_11_24/seed_40226 | None | Unknown | 0.5 | Nothing. |
| 2023_09_05_10_18 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_05_10_18/seed_40226 | None | Unknown | 0.0 | Nothing. |
| 2023_09_05_21_49 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_05_21_49/seed_40226 | None | Bad | 2.5 | Nothing. |
| 2023_09_07_10_25 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_07_10_25/seed_40226 | None | Bad | 3.0 | Nothing. |

Logs of the env 'DroneHoverBulletEnvWithoutAdversary':
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- | ----------- |----------- |
| 2023_09_12_11_23 | 44165 | DroneHoverBulletEnvWithoutAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithoutAdversary-v0/ppo/2023_09_12_11_23/seed_44165 | None | Good | 0.0 | Nothing. |


Logs of the env 'DroneHoverBulletEnvWithAdversaryInitial':
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- | ----------- |----------- |
| 2023_09_14_16_00 | 37007 | DroneHoverBulletEnvWithAdversaryInitial-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversaryInitial-v0/ppo/2023_09_14_16_00/seed_37007 | None | not work | 1.5 | Nothing. |
| 2023_09_15_13_42 | 49593 | DroneHoverBulletEnvWithAdversaryInitial-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversaryInitial-v0/ppo/2023_09_15_13_42/seed_49593 | None | not work | 1.0 | Nothing. |
| 2023_09_17_10_41 | 14929 | DroneHoverBulletEnvWithAdversaryInitial-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversaryInitial-v0/ppo/2023_09_17_10_41/seed_14929 | None | not work | 2.0 | Nothing. |
| 2023_09_25_14_10 | 63293 | DroneHoverBulletEnvWithAdversaryInitial-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithAdversaryInitial-v0/ppo/2023_09_25_14_10/seed_63293 | None | unknowning | 0.5 | Nothing. |


Logs of the env 'DroneHoverBulletEnvWithRandomHJAdversaryInitial':
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | episodic random uniform distb level |else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- | ----------- |----------- |
| 2023_10_11_12_04 | 61897 | DroneHoverBulletEnvWithRandomHJAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_11_12_04/seed_61897 | None | Unknown | step Boltzmann distb level | Nothing. |
| 2023_10_11_12_07 | 62086 | DroneHoverBulletEnvWithRandomHJAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_11_12_07/seed_62086 | None | Unknown | step random uniform distb level | Nothing. |
| 2023_10_11_12_33 | 63596 | DroneHoverBulletEnvWithRandomHJAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_11_12_33/seed_63596 | None | Unknown | episodic Boltzmann distb level | Nothing. |
| 2023_10_14_23_13 | 33562 | DroneHoverBulletEnvWithRandomHJAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_14_23_13/seed_33562 | None | Unknown | episodic Boltzmann distb level | Add implementations to the reset function |
| 2023_10_16_12_30 | 36701 | DroneHoverBulletEnvWithRandomHJAdversary-v0 | ppo | results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_16_12_30/seed_36701 | None | Unknown | episodic Boltzmann distb level | Add implementations to the reset function |
### 2. Test:
Notice: the environment in test also needs to change while we want to see different disturbance level.

1. Test with random disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithRandomAdversary-v0'`
2. Test without disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithoutAdversary-v0'`
3. Test without control inputs in the HJ disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversary-v0'  --nocontrol`

4. Test with different initial states env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversaryInitial-v0'`

5. Test with trained model in different envs and display:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversary-v0'`
   example: 
   `python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_08_31_11_48/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0'`

   original reward + with 17 input in 0 disturbance env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithoutAdversary-v0/ppo/2024_02_04_15_51/seed_08984 --env 'DroneHoverBulletEnvWithoutAdversary-v0'

   original reward + with 34 input in distb=1.5 env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_08_31_11_48/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0'

   original reward + with 34 input in distb=1.0 env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_03_12_11/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0'

   original reward + with 34 input in Boltzman distb env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_16_12_30/seed_36701 --env 'DroneHoverBulletEnvWithAdversary-v0'
   
   test reward + with 17 input in 0 disturbance env: 
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithoutAdversary-v0/ppo/2024_02_06_15_50/seed_50658 --env 'DroneHoverBulletEnvWithoutAdversary-v0'

   test reward + with 17 input in Boltzman distb env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2024_02_12_12_58/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0'

   test reward + with 17 input in Boltzman distb env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2024_02_12_12_58/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0' --test

   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2024_03_18_21_33/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0'  


   python -m phoenix_drone_simulation.play --ckpt training_results/boltzmann/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0' 
   
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithoutAdversary-v0/ppo/2024_02_06_15_50/seed_50658 --env 'DroneHoverBulletEnvWithAdversary-v0' --save

   python -m phoenix_drone_simulation.play --ckpt training_results/boltzmann/seed_40226/obs_noise_1backup --env 'DroneHoverBulletEnvWithAdversary-v0' --save
   
   training_results/boltzmann/seed_40226/obs_noise_1backup

6. Test with trained model in different envs and save the videos:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversary-v0'  --save`
   example: 
   `python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_08_31_11_48/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0' --save`
   ` python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletFreeEnvWithRandomHJAdversary-v0/ppo/2023_11_19_22_12/seed_63665 --env 'DroneHoverBulletFreeEnvWithoutAdversary-v0' --save `

   original reward + with 34 input in distb=1.0 env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_03_12_11/seed_40226 --env 'DroneHoverBulletEnvWithAdversary-v0' --save

   original reward + with 34 input in Boltzman distb env:
   python -m phoenix_drone_simulation.play --ckpt results_train_crazyflie/DroneHoverBulletEnvWithRandomHJAdversary-v0/ppo/2023_10_16_12_30/seed_36701 --env 'DroneHoverBulletEnvWithAdversary-v0' --save
   
### 3. Modifications of the environment
#### 3.1 Reward Function Design
Reward Formulation:
```
# Hanyang: the current valid rewards are: penalty_rpy, penalty_spin, penalty_velocity and penalty_terminal
penalty_rpy = self.penalty_angle * np.linalg.norm(self.drone.rpy - self.target_rpy)
penalty_spin = self.penalty_spin * np.linalg.norm(self.drone.rpy_dot - self.target_rpy_dot)
penalty_terminal = self.penalty_terminal if self.compute_done() else 0.  # Hanyang: try larger crash penalty
penalty_velocity = self.penalty_velocity * np.linalg.norm(self.drone.xyz_dot)
penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_spin,
                     penalty_velocity, penalty_action, penalty_terminal])

# L2 norm:
distance = self.penalty_z * np.linalg.norm(self.drone.xyz[2] - self.target_pos[2])
reward = -penalties - distance```

##### 3.1.1 Test 1
```
penalty_action: float = 0.,  # Hanyang: original is 1e-4
penalty_angle: float = 1e-2,  # Hanyang: original is 0
penalty_spin: float = 1e-2,  # Hanayng: original is 1e-4
penalty_terminal: float = 1000,  # Hanyang: try larger crash penalty,original is 100
penalty_velocity: float = 1e-2,  # Hanyang: original is 0
penalty_z: float = 1.0,  # Hanyang: original is 0
```
Performance:
distb = 0.0
| Trained Algorithm | move? | rise? | spin? | crash? |
| ------------|-----------|------------|-----------| ----------- |
| Boltzmann distb | Yes | Yes | Yes | Yes (small distb) |


#### 3.1.2 Test 2
```
penalty_action: float = 0.,  # Hanyang: original is 1e-4
penalty_angle: float = 1.0,  # Hanyang: original is 0
penalty_spin: float = 1.0,  # Hanayng: original is 1e-4
penalty_terminal: float = 1000,  # Hanyang: try larger crash penalty,original is 100
penalty_velocity: float = 1.0,  # Hanyang: original is 0
penalty_z: float = 1.0,  # Hanyang: original is 0
```
Performance:
distb = 0.0
| Trained Algorithm | move? | rise? | spin? | crash? |
| ------------|-----------|------------|-----------| ----------- |
| Boltzmann distb | Yes | Some yes | No | Little |
| distb=1.5 | \ | \ | \ | Yes |

#### 3.1.3 Test 3 failed
```
```
penalty_action: float = 0.,  # Hanyang: original is 1e-4
penalty_angle: float = 1.0,  # Hanyang: original is 0
penalty_spin: float = 1.0,  # Hanayng: original is 1e-4
penalty_terminal: float = 1000,  # Hanyang: try larger crash penalty,original is 100
penalty_velocity: float = 1.0,  # Hanyang: original is 0
penalty_z: float = 0.0,  # Hanyang: original is 0
```
Performance:
distb = 0.0
| Trained Algorithm | move? | rise? | spin? | crash? |
| ------------|-----------|------------|-----------| ----------- |
| Boltzmann distb | Yes | Yes!!!! | No | No |
| distb=1.5 | \ | \ | \ | Yes |
