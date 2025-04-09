"""
This file will contain the Proximal Policy Agent (PPO).  
    
At each transistion we need to collect:
- state
- action
- advantage : This will need to be computed
- rewards
- value estimates


"""


import os
import random
import gymnasium as gym
import ale_py
import numpy as np

# Hyperparamaters
# --------------------------------------------------------------------

class PPOAgent:
    
    def __init__ (self):
        
        env = gym.make("ALE/Boxing-v5")
        Q = []
        TOTAL_EPISODES = 256
        GAMMA = 0.99 
        CLIP_EPSILON = 0.2
        ENTROPY_COEF = 0.01
    
    def advantageFunction (rewards):
        pass