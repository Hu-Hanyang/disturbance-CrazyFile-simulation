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
Logs of method1:
| command | environment | algorithm | train logdir | test (log and command) | performance | else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- |
|
#### Method2 `python adversaryhover_phoenix.py`
Its default logdir is `train_results_phoenix`.
Logs of method2:
| command | environment | algorithm | train logdir | test (log and command) | performance | else | 
| ------------|-----------|------------|-----------| ----------- |----------- |----------- |
### Test: `python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT`.



