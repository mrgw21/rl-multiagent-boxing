import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import ppo
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T


env = gym.make("ALE/Boxing-v5")
to_grayscale = T.Grayscale(num_output_channels=1)

def gather_data(actor = None, critic = None):
    """Gathers data for a full episode"""
    cumulative_reward = 0
    reward_tracker = []
    
    agent = ppo.PPOAgent(actor, critic)
        
    for x in range(100):
        done = False
        state = env.reset()
        cumulative_reward = 0
        
        while not done:
            
            # Turns it into a tensor for neural net input
            state = agent.state_manipulation(to_grayscale, state)
                
            # Retrieves the action to be taken
            action, prob = agent.get_action(state)
            action = torch.tensor(action, dtype=torch.int64)
            
            
            # Executes function and retrieves information
            new_state, reward, done, trunc, info = env.step(action)
            
            # Updates the information for that timestamp into the information dictionary
            agent.updateInformation(state, reward, done, trunc, info, action, prob)
            
            state = new_state
            
        cumulative_reward = agent.access_cumulative_reward()
        reward_tracker.append(cumulative_reward)
        agent.learn()
    
    # Saving model once complete
    agent.actor.save_model()
    agent.critic.save_model()
    output_to_excel(reward_tracker, "training_rewards.xlsx")

    
def evaluate (actor: str, critic: str):
    """Passes actor and critic models to evaluate agent"""
    agent = ppo.PPOAgent(actor, critic)
    
    cumulative_reward = 0
    reward_tracker = []
    agent = ppo.PPOAgent()
    for x in range(500):
        done = False
        state = env.reset()
        cumulative_reward = 0
        
        while not done:
            
            # Turns it into a tensor for neural net input
            state = agent.state_manipulation(to_grayscale, state)
                
            # Retrieves the action to be taken
            action, prob = agent.get_action(state)
            action = torch.tensor(action, dtype=torch.int64)
            
            
            # Executes function and retrieves information
            new_state, reward, done, trunc, info = env.step(action)
            
            # Updates the information for that timestamp into the information dictionary
            agent.updateInformation(state, reward, done, trunc, info, action, prob)
            
            state = new_state
            
        cumulative_reward = agent.access_cumulative_reward()
        reward_tracker.append(cumulative_reward)
    
    output_to_excel(reward_tracker, "eval_rewards.xlsx")
    
            

def output_to_excel(info: list, pathname = 'rewards.xlsx'):
    df = pd.DataFrame(info, columns=['Values'])
    df.to_excel(pathname, index=False, sheet_name='sheet1')

if __name__ == "__main__":
    gather_data()