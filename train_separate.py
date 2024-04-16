"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Hanyang Hu 
Date: 2024-04-06
Location: SFU Mars Lab
Status: Not finished
"""
import os
import time
import argparse
# from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.algs.model_separate import ModelS
from phoenix_drone_simulation.utils.loggers import setup_separate_logger_kwargs
from phoenix_drone_simulation.envs.hover_distb import DroneHoverFixedDistbEnv, DroneHoverBoltzmannDistbEnv, DroneHoverNoDistbEnv, DroneHoverRandomDistbEnv


def start_training(distb_type, distb_level, random_seed, algo):
    
    # random_seed = int(time.time()) % 2 ** 16   # 40226 
    
    # === Set up the environment, saving directory, and loggers ===
    if distb_type == "fixed" or None:
        env = DroneHoverFixedDistbEnv(distb_level=distb_level)
        default_log_dir = os.path.join('training_results/' + 'fixed'+'-'+f'distb_level_{distb_level}', 'seed_'+f"{random_seed}") 
        
    elif distb_type == "boltzmann":
        env = DroneHoverBoltzmannDistbEnv()
        default_log_dir = os.path.join('training_results/' + distb_type, 'seed_'+f"{random_seed}") 
        
    elif distb_type == "random":
        # Hanyang: not finished yet
        env = DroneHoverRandomDistbEnv()
        default_log_dir = os.path.join('training_results/' + distb_type, 'seed_'+f"{random_seed}")
        
    else:
        print("The required disturbance type is not supported yet. \n")
    
    if not os.path.exists(default_log_dir):
        os.makedirs(default_log_dir+'/')
    
    # === Set up the training model ===
    model = ModelS(
        alg=algo, 
        env = env,
        distb_type=distb_type,
        distb_level=distb_level,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )
    model.compile()  # set up the logger and the parallelized environment

    # === Train the model ===
    start_time = time.perf_counter()
    
    model.fit(epochs=301)    
    
    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--distb_type',         default="boltzmann",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--seed',               default=40226,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--algo',         default="ppo_separate",      type=str,           help='Type of training algorithms (default: "ppo_separate")', metavar='')
    
    args = parser.parse_args()

    algorithm = 'ppo_separate'

    start_training(distb_type=args.distb_type, distb_level=args.distb_level, random_seed=args.seed, algo=args.algo)
