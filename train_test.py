"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Hanyang Hu 
Date: 2024-04-06
Location: SFU Mars Lab
Status: Not finished
"""
import time
import argparse
# from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.algs.model_separate import ModelS
from phoenix_drone_simulation.utils.loggers import setup_logger_kwargs
from phoenix_drone_simulation.envs.hover_distb import DroneHoverFixedDistbEnv, DroneHoverBoltzmannDistbEnv, DroneHoverNoDistbEnv, DroneHoverRandomDistbEnv


def start_training(distb_type="fixed", distb_level=0.0):
    
    # === Set up the environment ===
    if distb_type == "fixed":
        env = DroneHoverFixedDistbEnv(distb_level=distb_level)
    elif distb_type == "boltzmann":
        env = DroneHoverBoltzmannDistbEnv()
    elif distb_type == "random":
        # Hanyang: not finished yet
        env = DroneHoverRandomDistbEnv()
    else:
        print("The required disturbance type is not supported yet. \n")
    
    # === Set up the results folder ===
    
    # === Set up the logger ===
    default_log_dir = f"results_train_crazyflie"
    logger_kwargs = setup_logger_kwargs(base_dir=default_log_dir, exp_name=exp_name, seed=_seed)
    
    # === Set up the algorithm ===
    
    

    # Create a seed for the random number generator
    # random_seed = int(time.time()) % 2 ** 16   # 40226 
    random_seed = 40226


    default_log_dir = f"results_train_crazyflie"

    # NEW: use algorithms implemented in phoenix_drone_simulation:
    # 1) Setup learning model
    model = Model(
        alg=algo, 
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )
    model.compile()  # set up the logger and the parallelized environment

    start_time = time.perf_counter()

    # 2) Train model - it takes typically at least 100 epochs for training
    model.fit(epochs=301)

    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
    # 3) Benchmark the f
    # inal policy and save results into `returns.csv`
    model.eval()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--task',               default="train",      type=str,           help='Select whether to train or test with render')
    parser.add_argument('--distb_type',         default="fixed",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--seed',               default=40226,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--multiagent',         default=False,        type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--settings',           default="training_fixed.json",        type=str,           help='The path to the training settings file (default: None)', metavar='')
    parser.add_argument('--test_distb_type',    default="fixed",      type=str,           help='Type of disturbance in the test environment', metavar='')
    parser.add_argument('--test_distb_level',   default=0.0,          type=float,         help='Level of disturbance in the test environment', metavar='')
    parser.add_argument('--max_test_steps',     default=500,          type=int,           help='Maximum number of steps in the test environment', metavar='')
    parser.add_argument('--num_videos',         default=2,            type=int,           help='Number of videos to generate in the test environment', metavar='')
    parser.add_argument('--fps',                default=50,           type=int,           help='Frames per second in the generated videos', metavar='')
    
    args = parser.parse_args()

    algorithm = 'ppo'
    env_id = 'DroneHoverBulletEnvWithRandomHJAdversary-v0'
    # env_id = 'DroneHoverCurriculumEnv-v0'

    # env_id = 'DroneHoverBulletFreeEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithRandomHJAdversary-v0'
    start_training(algo=algorithm, env_id=env_id)