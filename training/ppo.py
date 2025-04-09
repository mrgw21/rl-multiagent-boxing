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

# Hyperparamaters
# --------------------------------------------------------------------

class PPOAgent:
    
    def __init__ (self):
        
        GAMMA = 0.99 
        CLIP_EPSILON = 0.2
        ENTROPY_COEF = 0.01
        
        
        self.information = {
                'state': [],
                'state_value_function': [],
                'action': [],
                'log_prob_action': [],
                'reward': [],
                'cumulative_reward' : 0
            }
    
    def updateInformation (self, new_state, reward, done, trunc, info, action):
        self.information['state'].append(new_state)
        self.information['state_value_function'].append('NN_state')
        self.information['action'].append(action)
        self.information['log_prob_action'].append('log this action')
        self.information['reward'].append(reward)
        self.information['cumulative_reward'] += reward
        
    def reset_information (self):
                self.information = {
                'state': [],
                'state_value_function': [],
                'action': [],
                'log_prob_action': [],
                'reward': [],
                'cumulative_reward' : 0
            }
    
    def log_prob (self, action_probability):
        return 
                
    