"""
This script extends the adversary-based drone and its physics based on the original phoenix_simulation

Author: Hanyang Hu, Xilun Joe Zhang
Date: 2024-2-19
Location: SFU Mars Lab, CMU Safe AI Lab
"""

import time
import torch
import json
import numpy as np
from phoenix_drone_simulation.algs.model_curriculum import Curriculum_Model

"""
main idea:
1. initialize the model class
2. train with the first distb level
3. save the model and instantiate the model class with the saved model parameters

"""



def start_curriculum_training(algo, env_id, epochs=250):

    # Create a seed for the random number generator
    random_seed = int(time.time()) % 2 ** 16   # 40226 
    # random_seed = 40226

    # I usually save my results into the following directory:
    default_log_dir = f"results_curriculum_crazyflie"
    
    distb_levels = np.arange(0.0, 2.1, 0.1)
    
    training_time = []
    
    
    # #Test
    # distb_level = 0.0
    # model = Curriculum_Model(
    #             alg=algo, 
    #             env_id=env_id,
    #             log_dir=default_log_dir,
    #             init_seed=random_seed,
    #             distb_level=distb_level
    #         )

    # print(f"Try the  saving function here")
    # torch.save(model.actor_critic, f'{default_log_dir}/curriculum_models/model_{distb_level}.pth') 
    # # torch.save(model.actor_critic.state_dict(), f'{default_log_dir}/curriculum_models/model_{distb_level}.pth')
    # print("The model is saved.")
    
    # model = Curriculum_Model(
    #             alg=algo, 
    #             env_id=env_id,
    #             log_dir=default_log_dir,
    #             init_seed=random_seed,
    #             distb_level=0.1
    #         )
    # model.actor_critic = torch.load(f'{default_log_dir}/curriculum_models/model_{distb_level-0.1}.pth')
    # # model.actor_critic = torch.load(f'{default_log_dir}/curriculum_models/model_{distb_level-0.1}.pth')
    # print("The model have been loaded successfully.")
    
    for distb_level in distb_levels:
        
        if distb_level == 0.0:
            model = Curriculum_Model(
                alg=algo, 
                env_id=env_id,
                log_dir=default_log_dir,
                init_seed=random_seed,
                distb_level=distb_level
            )

        else:
            model = Curriculum_Model(
                alg=algo, 
                env_id=env_id,
                log_dir=default_log_dir,
                init_seed=random_seed,
                distb_level=distb_level
            )
            model.actor_critic = torch.load(f'{default_log_dir}/curriculum_models/model_{(distb_level-0.1):.1f}.pth')
            # model.actor_critic.load_state_dict(torch.load(f'{default_log_dir}/curriculum_models/model_{distb_level-0.1}.pth'))

        model.compile()  # set up the logger and the parallelized environment

        # 2) Train model - it takes typically at least 100 epochs for training
        start_time = time.perf_counter()
    
        model.fit(epochs=epochs)  #TODO: how to select the number of epochs?

        duration = time.perf_counter() - start_time
        # print(f"The time of training is {duration//3600}hours-{(duration%3600)//60}minutes-{(duration%3600)%60}seconds. \n")
        training_time.append(duration)
        
        torch.save(model.actor_critic, f'{default_log_dir}/curriculum_models/model_{distb_level:.1f}.pth') 
        # torch.save(model.actor_critic.state_dict(), f'{default_log_dir}/curriculum_models/model_{distb_level}.pth')

    # Save the list to a file
    with open(f'{default_log_dir}/trainning_time_{epochs}.json', 'w') as f:
        json.dump(training_time, f)
    # model.eval()
    
if __name__ == "__main__":
    algorithm = 'ppo'

    env_id = 'DroneHoverCurriculumEnv-v0'

    start_curriculum_training(algo=algorithm, env_id=env_id)



