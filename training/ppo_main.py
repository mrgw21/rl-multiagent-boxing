import os
import random
import gymnasium as gym
import ale_py
import numpy as np



env = gym.make("ALE/Boxing-v5", render_mode="human")
env.reset()
done = False
for x in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, trunc, info = env.step(action)
    print(x)
    
#actor = neural_ne.Actor
#critic = neural_ne.Critic


def gather_data():

    for x in range(100):
        
        done = False
        state = env.reset()
        
        while not done:
    
            information = {
                'state': [],
                'state_value_function': [],
                'action': [],
                'log_prob_action': [],
                'reward': [],
                'cumulative_reward' : 0
            }
            
            done = False
            while not done:
            
                # Retrieves the action to be taken
                #NN_action = actor.compute()
                
                # Computes the value function of the state
                #NN_state = critic.compute()
                
                new_state, reward, done, trunc, info = env.step(action)
                
                # Updates the information for that timestamp into the information dictionary
                information['state'].append(state)
                information['state_value_function'].append('NN_state')
                information['action'].append(action)
                information['log_prob_action'].append('log this action')
                information['reward'].append(reward)
                information['cumulative_reward'] += reward
                
                state = new_state
            
            
            
            
            
            
            