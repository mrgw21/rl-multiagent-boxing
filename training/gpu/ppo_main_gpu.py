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
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA device available.")

# Crucial that gymnasium is consistently used, rather tahn gym. Note that gymnasium is shortened to gym in import
env = gym.make("ALE/Boxing-v5", frameskip=1)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, grayscale_newaxis=True, screen_size=84, scale_obs=False, terminal_on_life_loss=True)
env = gym.wrappers.FrameStackObservation(env, 4)


def gather_data(actor=None, critic=None, target_episodes=2000, n_steps = 100):
    """Gathers data for multiple episodes, learning after each episode."""
    cumulative_reward = 0
    reward_tracker = []

    agent = ppo_gpu.PPOAgent(actor, critic)
    episode_num = 0

    os.makedirs("/mnt/saved_models", exist_ok=True)
    os.makedirs("/mnt/saved_metrics", exist_ok=True)

    while episode_num < target_episodes:
        done = False
        state = env.reset()
        episode_reward = 0
        episode_timestamp = 0
        while not done:
            state_t = agent.state_manipulation(state)

            action, prob = agent.get_action(state_t)
            action = torch.tensor(action, dtype=torch.int64)

            new_state, reward, done, trunc, info = env.step(action)

            agent.updateInformation(state_t, reward, done, trunc, info, action, prob)

            episode_reward += reward
            state = new_state
            
            episode_timestamp += 1
            
            if episode_timestamp % n_steps == 0:
                agent.add_final_state_value(state)
                agent.learn()

        reward_tracker.append(episode_reward)
        episode_num += 1
        print(f"Episode {episode_num} | Reward: {episode_reward}")

        agent.add_final_state_value(state)
        agent.learn()

        if episode_num % 100 == 0:
            agent.actor.save_model(f"/mnt/saved_models/actor_ep{episode_num}.pth")
            agent.critic.save_model(f"/mnt/saved_models/critic_ep{episode_num}.pth")
            output_to_excel(reward_tracker, f"/mnt/saved_metrics/training_rewards_{episode_num}.xlsx")
            print(f"Saved models at episode {episode_num}")

    agent.actor.save_model("/mnt/saved_models/actor_final.pth")
    agent.critic.save_model("/mnt/saved_models/critic_final.pth")
    output_to_excel(reward_tracker, "/mnt/saved_metrics/training_rewards_final.xlsx")



def evaluate(actor_path, critic_path):
    agent = PPOAgent(actor=actor_path, critic=critic_path)
    reward_tracker = []

    for episode in range(500):
        done = False
        state = env.reset()
        cumulative_reward = 0

        while not done:
            state_t = agent.state_manipulation(state)
            action, prob = agent.get_action(state_t, evaluate=True)
            action_tensor = torch.tensor(action, dtype=torch.int64, device=device)
            new_state, reward, done, trunc, info = env.step(action)
            agent.updateInformation(state_t, reward, done, trunc, info, action_tensor, prob)
            state = new_state
            cumulative_reward += reward

        reward_tracker.append(cumulative_reward)

    output_to_excel(reward_tracker, "eval_rewards.xlsx")

def output_to_excel(info: list, pathname='rewards.xlsx'):
    df = pd.DataFrame(info, columns=['Rewards'])
    df.to_excel(pathname, index=False, sheet_name='sheet1')

if __name__ == "__main__":
    gather_data()
    # For evaluation, call: evaluate('path_to_actor.pth', 'path_to_critic.pth')
