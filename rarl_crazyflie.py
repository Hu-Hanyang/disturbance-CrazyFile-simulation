"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Xilun Joe Zhang, Hanyang Hu
Date: 2024-2-18
Location: CMU Safe AI Lab, SFU Mars Lab
"""
import time
from phoenix_drone_simulation.algs.model_rarl import RARL_Model



def start_rarl_training(algo, protagonist_id, adversary_id, training_iterations=10, protagonist_iterations=101, adversary_iterations=101):
    
    random_seed = int(time.time()) % 2 ** 16   # 40226 
    default_log_dir = f"results_rarl_crazyflie"
    
    #TODO: the model initialization should be a torch model rather than only None
    protagonist_policy = None
    adversary_policy = None

    for _ in range(training_iterations):

    # 1) Setup learning model
        protagonist_model = RARL_Model(
            alg=algo, 
            env_id=protagonist_id,
            log_dir=default_log_dir,
            init_seed=random_seed,
            adversary_model = adversary_policy,
        )
        protagonist_model.compile()

        protagonist_start_time = time.perf_counter()

        # 2) Train model - it takes typically at least 100 epochs for training
        protagonist_policy, _ = protagonist_model.fit(epochs=protagonist_iterations)

        protagonist_duration = time.perf_counter() - protagonist_start_time
        print(f"The time for protagonist training  is {protagonist_duration//3600}hours-{(protagonist_duration%3600)//60}minutes-{(protagonist_duration%3600)%60}seconds. \n")
        
        protagonist_model.eval()
        
        # traning the adversary
        adversary_model = RARL_Model(
            alg=algo,  # choose between: trpo, ppo
            env_id=adversary_id,
            log_dir=default_log_dir,
            init_seed=random_seed,
            adversary_model = protagonist_policy,
        )
        adversary_model.compile()
        
        adversary_start_time = time.perf_counter()
        # 2) Train model - it takes typically at least 100 epochs for training
        adversary_policy, _ = adversary_model.fit(epochs=adversary_iterations)
        
        duration = time.perf_counter() - adversary_start_time
        print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
        
        adversary_model.eval()
    
    
    
if __name__ == "__main__":
    algorithm = 'ppo'
    # env_id = 'DroneHoverBulletEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletEnvWithAdversaryInitial-v0'
    protagonist_id = 'Drone_Hover_Protagonist-v0'
    adversary_id = 'Drone_Hover_Adversary-v0'

    # env_id = 'DroneHoverBulletFreeEnvWithAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithoutAdversary-v0'
    # env_id = 'DroneHoverBulletFreeEnvWithRandomHJAdversary-v0'
    training_iterations = 10
    start_rarl_training(algo=algorithm, protagonist_id=protagonist_id, adversary_id=adversary_id, training_iterations=training_iterations)