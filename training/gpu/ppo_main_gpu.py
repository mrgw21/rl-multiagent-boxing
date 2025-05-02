import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import ppo_gpu

from ppo_gpu import PPOAgent
from neural_ne_gpu import device

# --- Print CUDA info just once at startup ---
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES")) # Print the CUDA visible devices
print("torch.cuda.device_count() =", torch.cuda.device_count()) # Print the number of CUDA devices
if torch.cuda.is_available(): # If CUDA is available 
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}") # Print the name of the CUDA device
else:
    print("No CUDA device available.") # Print that no CUDA device is available

# Crucial that gymnasium is consistently used, rather tahn gym. Note that gymnasium is shortened to gym in import
env = gym.make("ALE/Boxing-v5", frameskip = 1)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, grayscale_newaxis=True, screen_size=84, scale_obs=False, terminal_on_life_loss=True)
env = gym.wrappers.FrameStackObservation(env, 4)


def gather_data(actor=None, critic=None, target_episodes=3000):
    """
    Gathers data for multiple episodes, learning after each episode.
    Args:
        actor (str): The path to the actor model.
        critic (str): The path to the critic model.
        target_episodes (int): The number of episodes to run.
    Outputs:
        training_rewards.xlsx: The rewards for the training episodes.
        saved_models: The saved models.
        saved_metrics: The saved metrics.
    """
    cumulative_reward = 0
    reward_tracker = []
    best_reward = float('-inf')

    agent = ppo_gpu.PPOAgent(actor, critic)
    episode_num = 0

    os.makedirs("/mnt/new_saved_model", exist_ok=True)
    os.makedirs("/mnt/new_saved_models", exist_ok=True)
    os.makedirs("/mnt/new_saved_metrics", exist_ok=True)

    while episode_num < target_episodes:
        done = False
        state = env.reset()
        episode_reward = 0
        episode_timestamp = 0
        while not done:
            state_t = agent.state_manipulation(state)

            action, prob = agent.get_action(state_t) # Get the action and the probability of the action
            action = torch.tensor(action, dtype=torch.int64) # Convert the action to a tensor

            new_state, reward, done, trunc, info = env.step(action)

            agent.updateInformation(state_t, reward, done, trunc, info, action, prob) # Update the information

            episode_reward += reward # Update the episode reward
            state = new_state # Update the state
            
            episode_timestamp += 1

        reward_tracker.append(episode_reward)
        episode_num += 1
        print(f"Episode {episode_num} | Reward: {episode_reward}") # Print the episode number and the reward

        agent.add_final_state_value(state) # Add the final state value
        agent.learn() # Learn from the collected data
        
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.actor.save_model("/mnt/new_saved_model/highest_model.pth") # Save the highest reward model
            print(f"Saving current best model with {best_reward} reward")
            
        if episode_num % 100 == 0:
            agent.actor.save_model(f"/mnt/new_saved_models/actor_ep{episode_num}.pth") # Save the actor model at the current episode
            agent.critic.save_model(f"/mnt/new_saved_models/critic_ep{episode_num}.pth") # Save the critic model at the current episode
            output_to_excel_reward(reward_tracker, f"/mnt/new_saved_metrics/training_rewards_{episode_num}.xlsx") # Save the rewards to an excel file
            print(f"Saved models at episode {episode_num}")

    output_to_excel(agent.loss_tracker, "/mnt/new_saved_metrics/loss.xlsx") # Output the loss to an excel file
    agent.actor.save_model("/mnt/new_saved_models/actor_final.pth") # Save the final actor model
    agent.critic.save_model("/mnt/new_saved_models/critic_final.pth") # Save the final critic model
    output_to_excel_reward(reward_tracker, "/mnt/new_saved_metrics/training_rewards_final.xlsx") # Save the final rewards



def evaluate(actor_path, critic_path):
    """
    Evaluates the model.
    Args:
        actor_path (str): The path to the actor model.
        critic_path (str): The path to the critic model.
    Outputs:
        eval_rewards.xlsx: The rewards for the evaluation episodes.
    """
    agent = PPOAgent(actor=actor_path, critic=critic_path)
    reward_tracker = []

    for episode in range(500):
        done = False
        state = env.reset()
        cumulative_reward = 0

        while not done:
            state_t = agent.state_manipulation(state) # Manipulate the state
            action, prob = agent.get_action(state_t, evaluate=True) # Get the action and the probability of the action
            action_tensor = torch.tensor(action, dtype=torch.int64, device=device) # Convert the action to a tensor
            new_state, reward, done, trunc, info = env.step(action) # Step the environment
            agent.updateInformation(state_t, reward, done, trunc, info, action_tensor, prob) # Update the information
            state = new_state # Update the state
            cumulative_reward += reward

        reward_tracker.append(cumulative_reward) 

    output_to_excel_reward(reward_tracker, "eval_rewards.xlsx") # Output the rewards to an excel file

def output_to_excel(reward_list, filename):
    """
    Outputs the loss to an excel file.
    Args:
        reward_list (list): The rewards.
        filename (str): The path to the excel file.
    Outputs:
        rewards.xlsx: The rewards.
    """
    df = pd.DataFrame({'Episode': list(range(1, len(reward_list)+1)),
                       'Loss': reward_list})
    df.to_excel(filename, index=False)

def output_to_excel_reward(reward_list, filename):
    """
    Outputs the rewards to an excel file.
    Args:
        reward_list (list): The rewards.
        filename (str): The path to the excel file.
    Outputs:
        rewards.xlsx: The rewards.
    """
    df = pd.DataFrame({'Episode': list(range(1, len(reward_list)+1)),
                       'Reward': reward_list})
    df.to_excel(filename, index=False)

if __name__ == "__main__":
    gather_data()
    # For evaluation, call: evaluate('path_to_actor.pth', 'path_to_critic.pth')
