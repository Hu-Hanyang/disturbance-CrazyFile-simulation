"""Play and render a trained policy.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Added:      16.11.2021
Updated:    16.04.2022 Purged old function snipptes
Updated:    4.09.2023 add function 'play_without_control' and some notations by Hanyang
"""
import cv2
import gym
import time
import time
import argparse
import os
import torch
import numpy as np
import warnings
from gym.wrappers import Monitor

# local imports
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.mpi_tools import is_root_process

try:
    import pybullet_envs  # noqa
except ImportError:
    if is_root_process():
        warnings.warn('pybullet_envs package not found.')


def play_after_training(actor_critic, env, noise=False):
    if not noise:
        actor_critic.eval()  # Set in evaluation mode before playing
    i = 0
    # pb.setRealTimeSimulation(1)
    # env = Monitor(env, 'test_results_videos', force=True)

    while True:
        done = False
        env.render()
        x = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            env.render()
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            time.sleep(1./120)
        # Hanyang: for video recording
        # env.close()
        i += 1
        print(
            f'Episode {i}\t Return: {ret}\t Length: {episode_length}\t Costs:{costs}')


def random_play(env_id, use_graphics):
    env = gym.make(env_id)
    i = 0
    rets = []
    TARGET_FPS = 60
    target_dt = 1.0 / TARGET_FPS
    while True:
        i += 1
        done = False
        env.render(mode='human') if use_graphics else None
        env.reset()
        ts = time.time()
        ret = 0.
        costs = 0.
        ep_length = 0
        while not done:
            ts1 = time.time()
            if use_graphics:
                env.render()
                # time.sleep(0.00025)
            action = env.action_space.sample()
            _, r, done, info = env.step(action)
            ret += r
            ep_length += 1
            costs += info.get('cost', 0.)
            delta = time.time() - ts1
            if delta < target_dt:
                time.sleep(target_dt-delta)  # sleep delta time
            # print(f'FPS: {1/(time.time()-ts1):0.1f}')
        rets.append(ret)
        print(f'Episode {i}\t Return: {ret}\t Costs:{costs} Length: {ep_length}'
              f'\t RetMean:{np.mean(rets)}\t RetStd:{np.std(rets)}')
        print(f'Took: {time.time()-ts:0.2f}')

def play_without_control(actor_critic, env, noise=False):
    if not noise:
        actor_critic.eval()  # Set in evaluation mode before playing
    i = 0
    # pb.setRealTimeSimulation(1)
    while True:
        done = False
        env.render()
        x = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            env.render()
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = actor_critic(obs)
            action = np.zeros_like(action)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            time.sleep(1./120)  # 0.0083 second
        i += 1
        print(
            f'Episode {i}\t Return: {ret}\t Length: {episode_length}\t Costs:{costs}')

def save_videos(images, env, id):
    """Hanyang
    Input:
        images: list, a list contains a sequence of numpy ndarrays
        env: the quadrotor and task environment
    """
    # Define the output video parameters
    fps = 50  # Frames per second
    frame_width, frame_height = env.render_width, env.render_height
    save_path = 'test_results_videos'
    filename = f'episode{id+1}-{time.strftime("%Y_%m_%d_%H_%M")}.mp4'

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., 'XVID')
    out = cv2.VideoWriter(save_path+'/'+filename, fourcc, fps, (frame_width, frame_height))

    # Write frames to the video file
    for image in images:
        image = np.asarray(image, dtype=np.uint8)
        out.write(image)

    print(f"The video episode{id+1}.mp4 is saved at {save_path+'/'+filename}.")
    # Release the VideoWriter object
    out.release()

    # Destroy any OpenCV windows if they were opened
    cv2.destroyAllWindows()

def play_and_save(actor_critic, env, episodes=2, noise=False):
    # Hanyang: add function to save the images and generate videos, not finished, 2023.9.20
    if not noise:
        actor_critic.eval()  # Set in evaluation mode before playing
    episode = 0
    images = [ [] for _ in range(episodes)]
    # pb.setRealTimeSimulation(1)
    while episode < episodes:
        # env.render()
        done = False
        images[episode].append(env.capture_image()) # initial image
        x = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            # env.render()
            images[episode].append(env.capture_image())
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            time.sleep(1./120)  # 0.0083 second
        
        episode += 1
        print(f'Episode {episode}\t Return: {ret}\t Length: {episode_length}\t Costs:{costs}')
    
    for i in range(len(images)):
        save_videos(images[i], env, i)

    
if __name__ == '__main__':
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Choose from: {ppo, trpo}')
    parser.add_argument('--env', type=str,
                        help='Example: HopperBulletEnv-v0')
    parser.add_argument('--random', action='store_true',
                        help='Visualize agent with random actions.')
    parser.add_argument('--noise', action='store_true',
                        help='Visualize agent with random actions.')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering.')
    parser.add_argument('--nocontrol', action='store_true',
                        help='whether to add control inputs')
    parser.add_argument('--save', action='store_true',
                        help='whether to save the images and generate videos')
    args = parser.parse_args()
    env_id = None
    use_graphics = False if args.no_render else True

    if args.random:
        # play random policy
        assert env_id or hasattr(args, 'env'), 'Provide --ckpt or --env flag.'
        env_id = args.env if args.env else env_id
        random_play(env_id, use_graphics)

    elif args.nocontrol:
        # play no policy, that is to say, no control in the environment
        assert args.ckpt, 'Define a checkpoint for non-random play!'  # Hanyang: maybe not necessary?
        ac, _ = utils.load_actor_critic_and_env_from_disk(args.ckpt)
        env = gym.make(args.env)
        print("-"*150)
        print(f"The environment is {env} and no control inputs.")
        print("-"*150)

        play_without_control(
            actor_critic=ac,
            env=env,
            noise=args.noise
        )

    elif args.save:
        # save the images using render function
        assert args.ckpt, 'Define a checkpoint for non-random play!'  # Hanyang: maybe not necessary?
        ac, _ = utils.load_actor_critic_and_env_from_disk(args.ckpt)
        env = gym.make(args.env)
        print("-"*150)
        print(f"The environment is {env} and save the videos.")
        print("-"*150)

        play_and_save(actor_critic=ac, env=env, episodes=2, noise=args.noise)

    else:
        assert args.ckpt, 'Define a checkpoint for non-random play!'
        ac, _ = utils.load_actor_critic_and_env_from_disk(args.ckpt)
        env = gym.make(args.env)
        print("-"*150)
        print(f"The mode is playing after training! \n")
        print(f"The environment is {env} and load the trained model form {args.ckpt}.")
        print("-"*150)


        play_after_training(
            actor_critic=ac,
            env=env,
            noise=args.noise
        )