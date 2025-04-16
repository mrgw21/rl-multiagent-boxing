import numpy as np
import os
from pettingzoo.atari import boxing_v2
import ale_py
# if using gymnasium
import shimmy

import gymnasium as gym
import random

class Standard_Agent:
    '''
    
    '''

    #Method for defining global variables
    def __init__(self, name, render = None):
        #Hyperparams
        self.feature_length = 524800                        #Length of the features
        self.weights = np.zeros((18, self.feature_length))
        self.alpha = 0.2
        self.epsilon = 0.05
        self.gamma = 0.999
        self.episodes = 5000
        self.lamb = 0.5

        #Environment specific
        self.multiplayer_env = boxing_v2.parallel_env(obs_type = 'ram',render_mode=render)
        self.singleplayer_env = gym.make("ALE/Boxing-ram-v5", obs_type="ram",render_mode=render)
        self.ID = "first_0"

        #Recall params
        self.folder_path = "/agents/Saved Weights/" + name + ".txt"

    #Function designed to extract the list of features from the atari observation based on the type 'ram'
    def feature_extraction(self, observations):
        
        #Initial observations, which are stored as a list of 128 values, ranging from 0-256 are 'unpacked' to bits.
        binary_observations = np.unpackbits(observations)
        #Make use of the np.triu_indices function to return pairs, i and j, of all possible bit combinations, ensuring no repeats
        i, j = np.triu_indices(len(binary_observations), k=1)
        #Execute these combinations using the bianry operator '&'
        pairwise_ands = binary_observations[i] & binary_observations[j]
        #Concataenate the two into a single list
        features = np.concatenate([binary_observations, pairwise_ands])

        return features



    def model(self):
        pass

    def value(self, feature_space, action):
        return feature_space @ self.weights[action]

    def policy(self, action_space, features):
        value_hold = -np.inf
        action_chosen = 0
        actions = list(range(action_space))
        random.shuffle(actions)

        if np.random.rand() < self.epsilon:
            action_chosen = self.singleplayer_env.action_space.sample()
            value_hold = self.value(features, action_chosen)
        else:
            for action in actions:
                value = self.value(features, action)

                if value > value_hold:
                    action_chosen = action
                    value_hold = value
        
        return action_chosen, value_hold

    def single_learn_SARSA(self):
        returns = []
        for episode in range(self.episodes):
            e_traces = np.zeros((18,self.feature_length))
            sum_of_rewards = 0
            observations, _ = self.singleplayer_env.reset()

            features = self.feature_extraction(observations)
            action, value = self.policy(self.singleplayer_env.action_space.n, features)

            episode_over = False
            while not episode_over:
                observations, reward, terminated, truncated, _ = self.singleplayer_env.step(action)

                next_features = self.feature_extraction(observations)
                next_action, next_value = self.policy(self.singleplayer_env.action_space.n, next_features)

                episode_over = terminated or truncated

                e_traces *= self.gamma * self.lamb
                e_traces[action][features > 0] = 1

                delta = reward + self.gamma*next_value - value
                self.weights[action] += self.alpha * delta * e_traces[action]

                features, action, value = next_features, next_action, next_value

                sum_of_rewards += reward
            
            returns.append(sum_of_rewards)
            print(f"Episode - {episode}")
            print(sum_of_rewards)
            print("------")
            print("")
        
        np.savetxt(self.folder_path, self.weights)
        return returns

agent = Standard_Agent("1st Test - 15-04-2025")
weights, returns = agent.single_learn_SARSA()
print(returns[-100:])