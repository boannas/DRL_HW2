"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.SARSA import SARSA
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #


    import time
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    save_name = 'SARSA_rerun_df_09'
    log_dir = os.path.join("logs", args_cli.task, save_name)
    writer = SummaryWriter(log_dir)

    # hyperparameters
    num_of_action = 7
    action_range = [-10, 10]  # [min, max]
    discretize_state_weight = [5, 11, 3, 3]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 5000
    start_epsilon = 1.0
    epsilon_decay = 0.9988  # reduce the exploration over time
    final_epsilon = 0.01
    discount = 0.5

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "SARSA"
    agent = SARSA(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_reward = 0
    rewards_history = []
    training_times = []
    lengths_history = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0
                episode_length = 0
                start_time = time.time()
                
                
                # Initialize discrete...
                obs_dis = agent.discretize_state(obs)           
                action, action_idx = agent.get_action(obs)

                while not done:
                    # agent stepping
                    next_action, next_action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    cumulative_reward += reward_value
                    episode_length += 1
                    #######################################################
                    next_obs_dis = agent.discretize_state(obs)


                    agent.update(
                        obs_dis, action_idx, reward, next_obs_dis, next_action_idx, done
                    )

                    done = terminated or truncated
                    obs = next_obs

                    obs_dis = next_obs_dis
                    action = next_action
                    action_idx = next_action_idx
                    #######################################################



                # Log training stats
                writer.add_scalar("Rewards/Episode Reward", cumulative_reward, episode)
                writer.add_scalar("Epsilon/Episode Epsilon", agent.epsilon, episode)
                writer.add_scalar("Episode Length", episode_length, episode)
                writer.add_scalar("Training Time per Episode", time.time() - start_time, episode)
                writer.add_histogram("Actions/Action Distribution", np.array(action_idx), episode)
                writer.add_histogram("Q-Values/Per Action", np.array(list(agent.q_values.values())), episode)

####
                rewards_history.append(cumulative_reward)
                lengths_history.append(episode_length)
                training_times.append(time.time() - start_time)



                sum_reward += cumulative_reward
                if episode % 100 == 0:
                    avg_reward = sum_reward / 100.00
                    writer.add_scalar("Avg reward per 100 Episode", avg_reward, episode)


                    cumulative_reward = np.mean(rewards_history[-100:])
                    cumulative_episode_length = np.mean(lengths_history[-100:])
                    cumulative_training_time = np.mean(training_times[-100:])
    ###

                    writer.add_scalar("Cumsum/Total Reward", cumulative_reward, episode)
                    writer.add_scalar("Cumsum/Episode Length", cumulative_episode_length, episode)
                    writer.add_scalar("Cumsum/Training Time", cumulative_training_time, episode)

                    print("avg_score: ", sum_reward / 100.0)
                    sum_reward = 0
                    print(agent.epsilon)

                    # Save Q-Learning agent
                    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name, save_name)
                    agent.save_model(full_path, q_value_file)

                agent.decay_epsilon()
             
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        print("!!! Training is complete !!!")
        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()