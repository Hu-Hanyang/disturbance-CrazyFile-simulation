import time
import torch
from phoenix_drone_simulation.algs.model import Model


model = Model(
        alg=algo,  # choose between: trpo, ppo
        env_id=env_id,
        log_dir=default_log_dir,
        init_seed=random_seed,
    )

model.actor_critic = torch.load('model.pth')


model.compile()  # set up the logger and the parallelized environment

model.fit(epochs=301)
torch.save(model.actor_critic, 'model.pth')


model.actor_critic = torch.load('model.pth')

