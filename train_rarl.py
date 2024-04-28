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
import numpy as np
import psutil
import sys
from phoenix_drone_simulation.utils.mpi_tools import mpi_fork
# from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.algs.model_rarl import ModelS
from phoenix_drone_simulation.utils.loggers import setup_separate_logger_kwargs
from phoenix_drone_simulation.envs.hover_rarl import DroneHoverEnv, DroneAdvEnv 


def start_training(random_seed,algo):
    
    USE_CORES = 32
    
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))

    # Use number of physical cores as default. If also hardware threading CPUs
    # should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if USE_CORES > physical_cores else False
    if mpi_fork(USE_CORES, use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()
    
    ## initalize two training env 
    env = DroneHoverEnv(distb_level=0)
    env_adv = DroneAdvEnv(distb_level=0)
    
    
    default_log_dir_hover = os.path.join('training_results/' + 'Hover' + 'Multi-CORE', 'seed_'+f"{random_seed}")
    default_log_dir_adv = os.path.join('training_results/' + 'Adv' + 'Multi-CORE', 'seed_'+f"{random_seed}")
    
    if not os.path.exists(default_log_dir_hover):
        os.makedirs(default_log_dir_hover+'/')
    if not os.path.exists(default_log_dir_adv):
        os.makedirs(default_log_dir_adv+'/')
        
    env = DroneHoverEnv(distb_level=0)
    env_adv = DroneAdvEnv(distb_level=0)
    
    model_hover = ModelS(
        alg=algo, 
        env = env,
        log_dir=default_log_dir_hover,
        init_seed=random_seed,
        distb_type = "rarl",
        distb_level = 0,
        use_mpi=True,
    )
    model_hover.compile(num_cores=USE_CORES)  # set up the logger and the parallelized environment
    
    # set up training model adv
    model_adv = ModelS(
        alg=algo, 
        env = env_adv,
        log_dir=default_log_dir_adv,
        init_seed=random_seed,
        distb_type = "rarl",
        distb_level = 0,
        use_mpi=True,
    )
    model_adv.compile(num_cores=USE_CORES)  # set up the logger and the parallelized environment
    hover_policy, _ = model_hover.fit(epochs=50)
    dist_policy, _ = model_adv.fit(epochs=20)


    for _ in range(8):
        
        env = DroneHoverEnv(distb_level=0, adv_policy = dist_policy)
        env_adv = DroneAdvEnv(distb_level=0, env_policy = hover_policy)
        model_hover.compile(env=env, num_cores=USE_CORES)
        model_adv.compile(env=env_adv, num_cores=USE_CORES)
        # === Train the model ===
        start_time = time.perf_counter()
        hover_policy, _ = model_hover.fit(epochs=30)
        # === Train the model ===
        dist_policy, _ = model_adv.fit(epochs=30)
        duration = time.perf_counter() - start_time
        print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
        
    # # === Evaluate the model ===
    # model_hover._evaluate_model(dist_policy)
        
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--seed',               default=42,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--algo',         default="ppo_separate",      type=str,           help='Type of training algorithms (default: "ppo_separate")', metavar='')
    
    args = parser.parse_args()

    algorithm = 'ppo_separate'

    start_training(random_seed=args.seed, algo=args.algo)
