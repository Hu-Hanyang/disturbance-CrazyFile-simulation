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


def start_training(alg, distb_type, distb_level, random_seed):
    
    # random_seed = int(time.time()) % 2 ** 16   # 40226 
    
    # === Set up the environment, saving directory, and loggers ===
    if distb_type == "fixed" or None:
        env = DroneHoverFixedDistbEnv(distb_level=distb_level)
        obs_noise = env.observation_noise
        default_log_dir = os.path.join('training_results/' + 'fixed'+'-'+f'distb_level_{distb_level}', 'seed_'+f"{random_seed}", 'obs_noise_'+f"{obs_noise}") 
        
    elif distb_type == "boltzmann":
        env = DroneHoverBoltzmannDistbEnv()
        obs_noise = env.observation_noise
        default_log_dir = os.path.join('training_results/' + distb_type, 'seed_'+f"{random_seed}", 'obs_noise_'+f"{obs_noise}") 
        
    elif distb_type == "random":
        # Hanyang: not finished yet
        env = DroneHoverRandomDistbEnv()
        obs_noise = env.observation_noise
        default_log_dir = os.path.join('training_results/' + distb_type, 'seed_'+f"{random_seed}", 'obs_noise_'+f"{obs_noise}")
        
    else:
        print("The required disturbance type is not supported yet. \n")
    
    if not os.path.exists(default_log_dir):
        os.makedirs(default_log_dir+'/')
    
    # === Set up the training model ===
    default_kwargs = utils.get_separate_defaults_kwargs(alg=alg)
    algorithm_kwargs: dict = {}
    
    kwargs = default_kwargs.copy()
    kwargs['seed'] = random_seed
    kwargs['distb_type'] = distb_type
    kwargs['distb_level'] = distb_level
    kwargs.update(**algorithm_kwargs)
    logger_kwargs = None  # defined by compile (a specific seed might be passed)
    
    # compile
    num_cores=os.cpu_count()
    logger_kwargs = dict(log_dir=default_log_dir, level=1, use_tensor_board=True, verbose=True)
    compiled = True
    
    # === Train the model ===
    epochs = 1
    if epochs is None:
        epochs = kwargs.pop('epochs')
    else:
        kwargs.pop('epochs')  # pop to avoid double kwargs
    start_time = time.perf_counter()
    
    learn_func = utils.get_learn_function(alg)  # Hanyang: the learn_func is outside the class of the algorithm
    ac, training_env = learn_func(env=env, logger_kwargs=logger_kwargs, epochs=epochs,**kwargs)
    
    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--distb_type',         default="boltzmann",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--seed',               default=42,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--alg',         default="ppo_separate",      type=str,           help='Type of training algorithms (default: "ppo_separate")', metavar='')
    
    args = parser.parse_args()

    algorithm = 'ppo_separate'

    start_training(alg=args.alg, distb_type=args.distb_type, distb_level=args.distb_level, random_seed=args.seed)
