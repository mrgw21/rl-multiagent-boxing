"""
This file will contain the Proximal Policy Agent (PPO).  
    
At each transistion we need to collect:
- state
- action
- advantage : This will need to be computed
- rewards
- value estimates

"""

import math
import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import torch
import neural_ne

# Hyperparamaters
# --------------------------------------------------------------------

class PPOAgent:
    
    def __init__ (self):
        
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsi = 0.2
        self.entropy = 0.01
        self.actor = neural_ne.Actor()
        self.critic = neural_ne.Critic()
        
        self.information = {
                'state': [],
                'state_value_function': [],
                'action': [],
                'done': [],
                'log_prob_action': [],
                'reward': [],
                'cumulative_reward' : 0
            }
    
    def updateInformation (self, state, reward, done, trunc, info, action):
        """Updates information from the action taken e.g., new states, rewards"""
        
        self.information['state'].append(state)
        self.information['state_value_function'].append(self.critic.forward(state))
        self.information['done'].append(done)
        self.information['action'].append(action)
        self.information['log_prob_action'].append('log_prob(action)')
        self.information['reward'].append(reward)
        self.information['cumulative_reward'] += reward
        
    def reset_information (self):
        """Resets information when the learning experience is over, ready for another one."""
        
        self.information = {
        'state': [],
        'state_value_function': [],
        'action': [],
        'log_prob_action': [],
        'reward': [],
        'cumulative_reward' : 0
    }
    
    def log_prob(action_probability):
        return math.log(action_probability)
    
    def compute_advantages (self, timestamp, state_value_function):
        """Computes advantages the long way - probably won't be used now"""
        rewards_after_state = self.information['reward'][timestamp:]
        return (sum([rewards_after_state[x] ^ (x+1) for x in range(len(rewards_after_state))]) - state_value_function)
    
    # This needs some more work
    def compute_gen_advantage_estimation (self):
        """Returns the general advantage estimation"""
        
        gae = 0
        returns = []
        rewards = self.information['reward']
        state_value = self.information['state_value_function']
        done = self.information['done']
        mask = [1 if x is False else 0 for x in done]
        
        for i in range(len(rewards)-1,-1,-1):
            delta = rewards[i] + (self.gamma * mask[i] * state_value[i+1]) - state_value[i]
            gae = delta + (self.gamma * mask[i] * self.lam * gae)
            returns.insert(0, gae + state_value[i])
        
        adv = np.array(returns) - state_value[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    
    def clipping(self, advantage, old_probability, new_probability):
        """Calculates the clipped value"""
        ratio = new_probability/old_probability
        return min(torch.clamp(ratio * advantage), torch.clamp(ratio, 1-self.clip_epsi, 1+self.clip_epsi) * advantage)
    
    def get_action (self, state):
        """Gets action from actor NN"""
        return self.actor.forward(state)