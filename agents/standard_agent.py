import numpy as np
import os
from pettingzoo.atari import boxing_v2
import ale_py
# if using gymnasium
import shimmy
import time

import gymnasium as gym
import random
from agent_utils import load_agent, save_agent, plot_learning_curve, test_agent

class Standard_Agent:
    '''
    SARSA Lambda agent that can use either full or reduced feature representation
    '''

    #Method for defining global variables
    def __init__(self, name, reduced_feature=False, render=None):
        #Hyperparams
        self.reduced_feature = reduced_feature
        if self.reduced_feature:
            self.feature_length = 10
        else:
            self.feature_length = 524800  # Length of the features
        self.weights = np.zeros((18, self.feature_length))
        self.alpha = 0.2
        self.epsilon = 0.05
        self.gamma = 0.999
        self.episodes = 15000
        self.lamb = 0.5

        #Environment specific
        self.multiplayer_env = boxing_v2.parallel_env(obs_type='ram', render_mode=render)
        self.singleplayer_env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render)
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
    
    def reduced_feature_extraction(self, ram_data):
        # Include player + opponent positions 
        player_x = int(ram_data[32])
        player_y = int(ram_data[34])
        opponent_x = int(ram_data[33])
        opponent_y = int(ram_data[35])
        
        # Relative positions
        dx = opponent_x - player_x
        dy = opponent_y - player_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Try adding game state information
        player_score = ram_data[18]
        opponent_score = ram_data[19]
        timer = ram_data[59]
        
        features = np.array([player_x, player_y, opponent_x, opponent_y, 
                            dx, dy, distance,
                            player_score, opponent_score, timer])
        return features

    def value(self, feature_space, action):
        return np.dot(feature_space, self.weights[action])

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
            start_time = time.time()
            e_traces = np.zeros((18, self.feature_length))
            sum_of_rewards = 0
            observations, _ = self.singleplayer_env.reset()

            # Extract features according to the selected method
            if self.reduced_feature:
                features = self.reduced_feature_extraction(observations)
            else:
                features = self.feature_extraction(observations)
                
            action, value = self.policy(self.singleplayer_env.action_space.n, features)

            episode_over = False
            while not episode_over:
                observations, reward, terminated, truncated, _ = self.singleplayer_env.step(action)

                # Extract next features according to the selected method
                if self.reduced_feature:
                    next_features = self.reduced_feature_extraction(observations)
                else:
                    next_features = self.feature_extraction(observations)
                    
                next_action, next_value = self.policy(self.singleplayer_env.action_space.n, next_features)

                episode_over = terminated or truncated

                # Update eligibility traces - different approach based on feature type
                e_traces *= self.gamma * self.lamb
                
                if self.reduced_feature:
                    # For reduced features, directly add the feature values to traces
                    e_traces[action] += features  
                else:
                    # For binary features, set to 1 where features are active
                    e_traces[action][features > 0] = 1

                delta = reward + self.gamma * next_value - value
                self.weights[action] += self.alpha * delta * e_traces[action]

                features, action, value = next_features, next_action, next_value

                sum_of_rewards += reward
            
            returns.append(sum_of_rewards)
            end_time = time.time()
            total_time = end_time - start_time
            print("Time for episode", total_time)
            print(f"Episode - {episode}")
            print(sum_of_rewards)
            print("------")
            print("")
        
        np.savetxt(self.folder_path, self.weights)

        return self.weights, returns

# Create agent with reduced features enabled
agent = Standard_Agent("Reduced_Feature_Test_16-04-2025", reduced_feature=True)
weights, returns = agent.single_learn_SARSA()
print(returns[-100:])

save_agent(agent, "agents/saved_agents/standard_reduced.pkl")
loaded_agent = load_agent("agents/saved_agents/standard_reduced.pkl")
test_agent(loaded_agent)
