"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Xubo, Xilun Joe Zhang, Hanyang Hu
Date: 2022-11-23
Updat: 2023-9-12
Location: SFU Mars Lab
"""
import time
from phoenix_drone_simulation.algs.model import Model


def start_training(algo, env_id):
    env_id = env_id

    # Create a seed for the random number generator
    # random_seed = int(time.time()) % 2 ** 16   # 40226 
    random_seed = 40226

    # I usually save my results into the following directory:
    default_log_dir = f"train_results_phoenix"

    # NEW: use algorithms implemented in phoenix_drone_simulation:
    # 1) Setup learning model
    model = Model(
        alg=algo,  # choose between: trpo, ppo
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )
    model.compile()

    start_time = time.perf_counter()

    # 2) Train model - it takes typically at least 100 epochs for training
    model.fit(epochs=301)

    duration = time.perf_counter() - start_time
    print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
    # 3) Benchmark the f
    # inal policy and save results into `returns.csv`
    model.eval()
    
if __name__ == "__main__":
    algorithm = 'ppo'
    # env_id = 'DroneHoverBulletEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithAdversaryInitial-v0'
    env_id = 'DroneHoverBulletEnvWithRandomHJAdversary-v0'

    # env_id = 'DroneHoverBulletFreeEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithRandomHJAdversary-v0'
    start_training(algo=algorithm, env_id=env_id)