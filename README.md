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

# Working Logs
## Basic commands
### Training: 
#### Method1 
`python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnvWithAdversary-v0`
Its default logdir is `/localhome/hha160/projects/disturbance-CrazyFile-simulation/train_results/`.
∆ TODO: This method takes too many paraller, needs to be changed.
Logs of method1:
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |  else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- |----------- |----------- |
| | | DroneHoverBulletEnvWithAdversary-v0 | ppo |  | None | Unknown | 1.5 | Nothing. |
#### Method2 `python adversaryhover_phoenix.py`
Its default logdir is `train_results_phoenix`.
Logs of method2:
| time | seed | environment | algorithm | train logdir | test (log and command) | performance | distb level |else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- | ----------- |----------- |
| 2023_08_31_11_44 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_08_31_11_48/seed_40226 | None | Not bad | 1.5 | Nothing. |
| 2023_09_02_13_22 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_02_13_22/seed_40226 | None | Unknown | 2.0 | Nothing. |
| 2023_09_03_12_11 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/ | None | Unknown | 1.0 | Nothing. |
| 2023_09_04_11_24 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_04_11_24/seed_40226 | None | Unknown | 0.5 | Nothing. |
| 2023_09_05_10_18 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_05_10_18/seed_40226 | None | Unknown | 0.0 | Nothing. |
| 2023_09_05_21_49 | 40226 | DroneHoverBulletEnvWithAdversary-v0 | ppo | train_results_phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023_09_05_21_49/seed_40226 | None | Bad | 2.5 | Nothing. |
### Test:
Notice: the environment in test also needs to change while we want to see different disturbance level.
1. Test with HJ disturbance env:
    `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversary-v0'`
2. Test with random disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithRandomAdversary-v0'`
3. Test without disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithoutAdversary-v0'`
4. Test without control inputs in the HJ disturbance env:
   `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT --env 'DroneHoverBulletEnvWithAdversary-v0'  --control False`