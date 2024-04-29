import cv2
import time
import time
import argparse
import os
import torch
import imageio
import numpy as np

# local imports
from phoenix_drone_simulation.utils.utils import get_file_contents
from phoenix_drone_simulation.utils.mpi_tools import is_root_process
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.envs.hover_distb import DroneHoverFixedDistbEnv, DroneHoverBoltzmannDistbEnv, DroneHoverNoDistbEnv, DroneHoverRandomDistbEnv


def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def load_actor_critic(file_name_path: str) -> tuple:
    """Loads ac module from disk. (@Sven).

    Parameters
    ----------
    file_name_path

    Returns
    -------
    tuple
        holding (actor_critic, env)
    """
    config_file_path = os.path.join(file_name_path, 'config.json')
    env_file_path = os.path.join(file_name_path, 'env_config.json')
    conf = get_file_contents(config_file_path)
    env_conf = get_file_contents(env_file_path)
    
    # Generate training env
    distb_type = conf.get("distb_type")
    distb_level = conf.get("distb_level")

    if distb_type == "fixed" or "rarl":
        env = DroneHoverFixedDistbEnv(distb_level=distb_level)
        
    elif distb_type == "boltzmann":
        env = DroneHoverBoltzmannDistbEnv()
        
    elif distb_type == "random":
        # Hanyang: not finished yet
        env = DroneHoverRandomDistbEnv()
        
    else:
        print("The required env is not supported yet. \n")
        raise NotImplementedError
    
    
    # print(f"Current env is {env_id}.")
    alg = conf.get('alg', 'ppo_separate')
    env_distb = env_conf.get("disturbance_level")

    if alg == 'sac':
        from phoenix_drone_simulation.algs.sac.sac import MLPActorCritic
        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            ac_kwargs=conf['ac_kwargs']
        )
    elif alg == "ddpg":
        from phoenix_drone_simulation.algs.ddpg.ddpg import MLPActorCritic
        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            ac_kwargs=conf['ac_kwargs']
        )
    else:
        ac = core.ActorCritic(
            actor_type=conf['actor'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=conf['use_standardized_obs'],
            use_scaled_rewards=conf['use_reward_scaling'],
            use_shared_weights=False,
            ac_kwargs=conf['ac_kwargs']
        )
    
    # Select the latest trained model
    if len(os.listdir(os.path.join(file_name_path, 'torch_save'))) > 1:
        model_folder = os.listdir(os.path.join(file_name_path, 'torch_save'))
        model_files = [file for file in model_folder if file.startswith("model") and file.endswith(".pt")]
        max_suffix = max([int(f.split('model')[1].split('.pt')[0]) for f in model_files if f != 'model.pt'])
        model_path = os.path.join(file_name_path, 'torch_save', f'model{max_suffix}.pt')
    elif len(os.listdir(os.path.join(file_name_path, 'torch_save'))) == 1:
        model_path = os.path.join(file_name_path, 'torch_save', 'model.pt')
    else:
        print("No trained model!")

    # model_path = '/home/xilun/disturbance-CrazyFlie-simulation/hover_policy.pt'
    model_path = '/home/xilun/disturbance-CrazyFlie-simulation/training_results/HoverMulti-CORE/seed_42/torch_save/model29.pt'
    # model_path = '/home/xilun/disturbance-CrazyFlie-simulation/training_results/AdvMulti-CORE/seed_42/torch_save/model29.pt'
    # model_path = '/home/xilun/disturbance-CrazyFlie-simulation/training_results/HoverMulti-CORE/seed_42/model49.pt'
    # ac = torch.load(model_path)
    ac.load_state_dict(torch.load(model_path), strict=False)
    print(f'Successfully loaded model from: {model_path}')

    return ac, env, env_distb


# Function to create GIF
def generate_gifs(image_list, filename, duration=0.1):
    images = []
    for img in image_list:
        images.append(img.astype(np.uint8))  # Convert to uint8 for imageio
    imageio.mimsave(f'{filename}', images, duration=duration)


def generate_videos(images, env, id, foldername):
    """Hanyang
    Input:
        images: list, a list contains a sequence of numpy ndarrays
        env: the quadrotor and task environment
    """
    # Define the output video parameters
    fps = 50  # Frames per second
    frame_width, frame_height = env.render_width, env.render_height
    filename = f'episode{id+1}-{len(images)}.mp4'

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., 'XVID')
    out = cv2.VideoWriter(foldername+'/'+filename, fourcc, fps, (frame_width, frame_height))

    # Write frames to the video file
    for image in images:
        image = np.asarray(image, dtype=np.uint8)
        out.write(image)
    # Release the VideoWriter object
    out.release()


def play_and_save(actor_critic, env, episodes, foldername):
    # Hanyang: add function to save the images and generate videos, not finished, 2023.9.20
    actor_critic.eval()  # Set in evaluation mode before playing
    episode = 0
    images = [ [] for _ in range(episodes)]
    # pb.setRealTimeSimulation(1)
    print(f"[INFO] The test starts (saving). \n")
    # Hanayng: add random seed here, but it doesn't work here
    # np.random.seed(42) 
    while episode < episodes:
        # env.render()
        done = False
        images[episode].append(env.capture_image()) # initial image
        x = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        # while not done:
        while not done and episode_length < 500:
            # env.render()
            # print(f"Episode {episode}, Step {episode_length}")
            # print(f"Episode {episode}, Step {episode_length}, done: {done}")
            images[episode].append(env.capture_image())
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            time.sleep(1./120)  # 0.0083 second
        
        print(f'Episode {episode}\t Return: {ret}\t Length: {episode_length}\t Costs:{costs}')
        generate_videos(images[episode], env, episode, foldername)
        # create_gif(images[episode], f'{foldername}/episode{episode}-{episode_length}steps.gif', duration=0.01)
        episode += 1


def play_after_training(actor_critic, env):
    actor_critic.eval()  # Set in evaluation mode before playing
    i = 0
    print(f"[INFO] The test starts \n.")
    
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


def test(train_distb_type, train_distb_level, train_seed, obs_noise, test_distb_type, test_distb_level, num_videos, save):

    #### Load the trained model ###################################
    file_name_path = '/home/xilun/disturbance-CrazyFlie-simulation/training_results/AdvMulti-CORE/seed_42'
    ac, training_env, env_distb = load_actor_critic(file_name_path)
    
    #### Create the environment and make save path ################################
    if test_distb_type == 'fixed' or None:
        env = DroneHoverFixedDistbEnv(distb_level=test_distb_level)
        obs_noise = env.observation_noise
        foldername = os.path.join('test_results/' + 'fixed'+'-'+f'distb_level_{test_distb_level}', f'obs_noise{obs_noise}', 
                                  f'using-{train_distb_type}-distb_level_{train_distb_level}_model') 
    else:  # 'boltzmann', 'random', 'rarl', 'rarl-population'
        env = DroneHoverBoltzmannDistbEnv()
        foldername = os.path.join('test_results_/' + test_distb_type, f'obs_noise{obs_noise}', 
                                  f'using-{train_distb_type}-distb_level_{train_distb_level}_model')
    if not os.path.exists(foldername):
        os.makedirs(foldername+'/')
    
    print(f"[INFO] The test environment is with {test_distb_type} distb type and {test_distb_level} distb level.")
    print(f"[INFO] Save the test videos ( or GIFs) at: {foldername}")

    #### Play the trained model in the test environment ################################
    if not save:
        play_after_training(ac, env)
    else:
        play_and_save(ac, env, num_videos, foldername)



if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')

    parser.add_argument('--train_distb_type',         default="boltzmann",      type=str,           help='Type of disturbance to be applied to the drones [None, "fixed", "boltzmann", "random", "rarl", "rarl-population"] (default: "fixed")', metavar='')
    parser.add_argument('--train_distb_level',        default=0.0,          type=float,         help='Level of disturbance to be applied to the drones (default: 0.0)', metavar='')
    parser.add_argument('--train_seed',               default=42,        type=int,           help='Seed for the random number generator (default: 40226)', metavar='')
    parser.add_argument('--obs_noise',                default=0,          type=int,         help='Observation noise level (default: 0.0)', metavar='')
    parser.add_argument('--test_distb_type',    default="fixed",      type=str,           help='Type of disturbance in the test environment', metavar='')
    parser.add_argument('--test_distb_level',   default=0.3,          type=float,         help='Level of disturbance in the test environment', metavar='')
    parser.add_argument('--num_videos',         default=3,            type=int,           help='Number of videos to generate in the test environment', metavar='')
    parser.add_argument('--save',              default=True,         type=str2bool,          help='Save the videos or not', metavar='')
    args = parser.parse_args()

    test(train_distb_type=args.train_distb_type, train_distb_level=args.train_distb_level, 
         train_seed=args.train_seed, obs_noise=args.obs_noise, test_distb_type=args.test_distb_type, 
         test_distb_level=args.test_distb_level, num_videos=args.num_videos, save=args.save)