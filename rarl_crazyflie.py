"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Xubo, Xilun Joe Zhang, Hanyang Hu
Date: 2022-11-23
Updat: 2023-9-12
Location: SFU Mars Lab
"""
import time
from phoenix_drone_simulation.algs.model_rarl import Model



def start_training(algo, agent_id, adversary_id, training_iterations=1):
    
    random_seed = 40226
    default_log_dir = f"train_results_phoenix"
    
    agent_model = None
    adversary_model = None
    for _ in range(training_iterations):

    # NEW: use algorithms implemented in phoenix_drone_simulation:
    # 1) Setup learning model
        agent_model = Model(
            alg=algo,  # choose between: trpo, ppo
            env_id=agent_id,
            log_dir=default_log_dir,
            init_seed=random_seed,
            adversary_model = adversary_policy,
        )
        agent_model.compile()

        start_time = time.perf_counter()

        # 2) Train model - it takes typically at least 100 epochs for training
        agent_policy, _ = agent_model.fit(epochs=200)

        duration = time.perf_counter() - start_time
        print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
        # 3) Benchmark the f
        # inal policy and save results into `returns.csv`
        agent_model.eval()
        
        # traning the adversary
        adversary_model = Model(
            alg=algo,  # choose between: trpo, ppo
            env_id=adversary_id,
            log_dir=default_log_dir,
            init_seed=random_seed,
            adversary_model = agent_policy,
        )
        adversary_model.compile()
        start_time = time.perf_counter()
        # 2) Train model - it takes typically at least 100 epochs for training
        adversary_policy, _ = adversary_model.fit(epochs=200)
        duration = time.perf_counter() - start_time
        print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
        # 3) Benchmark the f
        # inal policy and save results into `returns.csv`
        adversary_model.eval()
    
    
    
if __name__ == "__main__":
    algorithm = 'ppo'
    # env_id = 'DroneHoverBulletEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithAdversaryInitial-v0'
    agent_id = 'Drone_Hover_Agent'
    adversary_id = 'Drone_Hover_Adversary'

    # env_id = 'DroneHoverBulletFreeEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithRandomHJAdversary-v0'
    training_iterations = 10
    start_training(algo=algorithm, agent_id=agent_id, adversary_id=adversary_id, training_iterations=training_iterations)