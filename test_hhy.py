import numpy as np
import pybullet as pb
from pyvirtualdisplay import Display
import gym
import time
import imageio
import torch
from phoenix_drone_simulation.algs.model import Model
from gym.wrappers import Monitor
from phoenix_drone_simulation.algs.ppo.ppo import ProximalPolicyOptimizationAlgorithm

# output_dir = data_dir/env-id/algo/YY-MM-DD_HH-MM-SS/seed[seed]
# fpath = 'torch_save'
# fpath = osp.join(self.log_dir, fpath)
# fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
# fname = osp.join(fpath, fname)
# relative path: runs/phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023-07-17__20-39-02/seed_02390/torch_save/model.pt
# 1) set up algorithm, environment id and the trained model
algo = 'ppo'
env_id = "DroneHoverBulletEnvWithAdversary-v0"
default_log_dir = f"./tests/phoenix/random_d"
ac_kwargs = {
        "pi":	{
            "activation":	"relu",
            "hidden_sizes":	[
                50,
                50
            ]
        },
        "val":	{
            "activation":	"tanh",
            "hidden_sizes":	[
                64,
                64
            ]
        }
    }

# if "Adversary" in env_id:
#     assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
#     register(id=env_id, entry_point="{}:{}".format(
#         DroneHoverBulletEnvWithAdversary.__module__, 
#         DroneHoverBulletEnvWithAdversary.__name__), 
#         max_episode_steps=500,)
    
model = Model(
    alg=algo,  # choose between: trpo, ppo
    env_id=env_id,
    log_dir=default_log_dir,
    init_seed=2390,
)
model.compile()

model_path = 'runs/original_ppo/our_distb/DroneHoverBulletEnvWithAdversary-v0/ppo/2023-08-25__23-35-14/seed_40226/torch_save/model.pt'


logger_kwargs = model.logger_kwargs
ppo = ProximalPolicyOptimizationAlgorithm(actor='mlp', ac_kwargs=ac_kwargs, env_id=env_id, epochs=100, logger_kwargs=logger_kwargs)
ppo.ac.load_state_dict(torch.load(model_path)) 
ppo.ac.eval()

print("The model is loaded successfully!")
# model.eval()

# 4) visualize trained PPO model
env = gym.make(env_id)

# video setting
# display = Display(visible=0, size=(1400, 900))
# display.start()
# video_filename = 'test_video.mp4'
# video_writter = imageio.get_writer(video_filename, fps=30)


# Important note: PyBullet necessitates to call env.render()
# before env.reset() to display the GUI!
env.render() 

while True:
    obs = env.reset()
    done = False
    while not done:
        # frame = env.render(mode='rgb_array') 
        # video_writter.append_data(frame)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action, value, logp = ppo.ac.step(obs)
        obs, reward, done, info = env.step(action)

        time.sleep(0.05)
        if done:
            obs = env.reset()