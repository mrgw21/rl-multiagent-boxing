import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import ppo


env = gym.make("ALE/Boxing-v5", render_mode="human")
env.reset()
done = False
for x in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, trunc, info = env.step(action)
    print(x)

agent = ppo.PPOAgent()

def gather_data():
    """Gathers data for a full episode"""
    done = False
    state = env.reset()
    agent = ppo.PPOAgent()

    while not done:
    
        # Retrieves the action to be taken
        action = agent.get_action(state)
        
        # Executes function and retrieves information
        new_state, reward, done, trunc, info = env.step(action)
        
        # Updates the information for that timestamp into the information dictionary
        agent.updateInformation(state, reward, done, trunc, info, action)
        
        state = new_state
            

         
            
            
            
            