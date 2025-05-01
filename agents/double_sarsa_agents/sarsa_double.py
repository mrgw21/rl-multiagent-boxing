import numpy as np
import gymnasium as gym
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
from pettingzoo.atari import boxing_v2
import ale_py
import shimmy
import time
import gymnasium as gym
import random
from utils.agent_utils import load_linear_agent, test_linear_agent, save_linear_agent, plot_rewards
from utils.lsh_script import LSH
from utils.PER import store_experience, prioritised_sample, update_priority_order


class Base_Double_Sarsa_Agent:
    """
    Base Double Sarsa Agent from which the other agents inherit from.
    """
    def __init__(self, name, render=None, feature_type="reduced_ram"):
        """
        feature_type can be either "reduced_ram", "full_ram" or "lsh"
        """
        # Determine feature type 
        self.feature_type = feature_type
        if self.feature_type == "reduced_ram":
            self.feature_length = 10
        elif self.feature_type == "full_ram":
            self.feature_length = 524800  # Length of the features
        elif self.feature_type == "lsh":
            self.lsh = LSH()
            self.feature_length = self.lsh.num_rand_bit_vecs * self.lsh.hash_table_size
        elif self.feature_type == "semi_reduced_ram":
            self.feature_length = 1052
        else:
            raise ValueError("Feature Type not recognised")
        self.normalise_features = False

        # Create env using ram observation type
        if self.feature_type == "lsh":
            self.env = gym.make("ALE/Boxing-ram-v5", obs_type="rgb", render_mode="rgb_array")
        else:
            self.env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render)
        
        # Store agent name and rewards
        self.name = name
        self.rewards_history = []
        
        # Actions for environment
        self.action_list = [i for i in range(18)]
        self.num_actions = len(self.action_list)

        # Create variables + define hyperparameters that will apply to all agents
        # Rather than use zeros for weights, start with a random initialisation
        self.weights_1 = np.random.uniform(-0.001, 0.001, (self.num_actions, self.feature_length))
        self.weights_2 = np.random.uniform(-0.001, 0.001, (self.num_actions, self.feature_length))
        # Add target networks for more stable learning 
        self.target_weights_1 = self.weights_1.copy()
        self.target_weights_2 = self.weights_2.copy()
        self.target_update_freq = 1000
        self.steps = 0
        self.alpha = 0.01
        self.epsilon = 0.5
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.alpha_decay = 0.999
        self.alpha_min = 0.001
        self.gamma = 0.9
        
        # Add additional experience replay with cache hyperparameters
        # Prioritised experience replay using Schaul et al. (2015) - Prioritised Experience Replay).
        self.max_capacity = 10000
        self.replay_buffer = deque(maxlen=self.max_capacity)
        self.priorities = deque(maxlen=self.max_capacity)
        self.batch_size = 32
        # "how much prioritisation is used" - when exp_alpha is 0, all samples have the same probability of being sampled
        self.exp_alpha = 0.6
        # Importance sampling exponent - prevents bias of sampling from only high priority samples
        self.initial_beta = 0.4
        self.exp_beta = 0.4
        self.exp_beta_increment = (1.0 - 0.4) / 50000

    def update_target_networks(self):
        """
        Updates the target network weights to match the current weights.
        """
        self.target_weights_1 = self.weights_1.copy()
        self.target_weights_2 = self.weights_2.copy()
    
    def feature_extraction(self, observations):
        # Determine feature type
        if self.feature_type == "reduced_ram":
            return self.reduced_feature_extraction_ram(observations)
        elif self.feature_type == "full_ram":
            return self.feature_extraction_ram(observations)
        elif self.feature_type == "lsh":
            screen = self.env.render()
            return self.lsh.feature_extraction_lsh(screen)
        elif self.feature_type == "semi_reduced_ram":
            return self.feature_extraction_semi_ram(observations)
        else:
            raise ValueError("Feature Type not recognised") # Won't actually be called 


    def feature_extraction_semi_ram(self, observations):
        player_x = int(observations[32])
        player_y = int(observations[34])
        opponent_x = int(observations[33])
        opponent_y = int(observations[35])
        dx = abs(opponent_x - player_x) 
        dy = abs(opponent_y - player_y)
        manhattan_distance = dx + dy

        x_dist = np.unpackbits(np.array([dx], dtype=np.uint8))
        y_dist = np.unpackbits(np.array([dy], dtype=np.uint8))  
        man_dist = np.unpackbits(np.array([manhattan_distance], dtype = np.uint8))

        binary_observations = np.unpackbits(observations)

        if self.normalise_features:
            binary_observations = binary_observations.astype(np.float32) / 1.0
            x_dist = x_dist.astype(np.float32) / 1.0
            y_dist = y_dist.astype(np.float32) / 1.0
            man_dist = man_dist.astype(np.float32) / 1.0

            logic_bits = np.array([
                                    player_x > opponent_x,
                                    player_y > opponent_y,
                                    manhattan_distance < 10,
                                    manhattan_distance < 20
                                ], dtype=np.float32)
        else:
            logic_bits = np.array([
                                player_x > opponent_x,
                                player_y > opponent_y,
                                manhattan_distance < 10,
                                manhattan_distance < 20
                            ], dtype=np.uint8)


        features = np.concatenate([binary_observations, x_dist, y_dist, man_dist, logic_bits])
        return features
    
    #Function designed to extract the list of features from the atari observation based on the type 'ram'
    def feature_extraction_ram(self, observations):
        #Initial observations, which are stored as a list of 128 values, ranging from 0-256 are 'unpacked' to bits.
        binary_observations = np.unpackbits(observations)
        #Make use of the np.triu_indices function to return pairs, i and j, of all possible bit combinations, ensuring no repeats
        i, j = np.triu_indices(len(binary_observations), k=1)
        #Execute these combinations using the bianry operator '&'
        pairwise_ands = binary_observations[i] & binary_observations[j]
        #Concataenate the two into a single list
        features = np.concatenate([binary_observations, pairwise_ands])

        return features


    def reduced_feature_extraction_ram(self, ram_data):
        """
        Extracts only the necessary features from the ram data
        """
        #  feature extraction with additional features
        # From https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        player_x = int(ram_data[32])
        player_y = int(ram_data[34])
        opponent_x = int(ram_data[33])
        opponent_y = int(ram_data[35])
        
        # Relative positions
        dx = opponent_x - player_x
        dy = opponent_y - player_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Extra information
        player_score = ram_data[18] 
        opponent_score = ram_data[19]
        timer = ram_data[17]
        
        features = np.array([player_x, player_y, opponent_x, opponent_y, 
                            dx, dy, distance, player_score, opponent_score, timer])
        
        # Normalise features - depending on feature, adjust how much the value is normalised
        if self.normalise_features:
            features[0:4] /= 160.0
            features[4:6] /= 140.0
            features[6] /= 180.0
            features[7:9] /= 100.0
            features[9] /= 255.0
        return features
    
    def value(self, feature_space, action, weights):
        """
        Return value of action in a given feature space - simple linear approximation using dot product
        """
        return np.dot(weights[action], feature_space)
    
    def policy(self, state, online=False):
        """
        Epsilon-greedy policy using average of the two weights - needed for double learning
        """
        if np.random.rand() < self.epsilon:
            # return self.env.action_space.sample() # Random action
            return random.choice(self.action_list)
        else:
            # Use average value from both weights to get action with highest q value
            q_values = np.zeros(self.num_actions)
            if online:
                for action in self.action_list:
                    q_values[action] = (self.value(state, action, self.weights_1) + 
                                        self.value(state, action, self.weights_2)) / 2 
            else:
                for action in self.action_list:
                    q_values[action] = self.value(state, action, self.weights_1) 
            return np.argmax(q_values)
    
    
    def update_weights(self, action, state, td_error, update_weights):
        """Performs update directly to agent's weight vectors rather than to local weights"""
        update = self.alpha * td_error * state
        update = np.clip(update, -0.1, 0.1)

        if update_weights is self.weights_1:
            self.weights_1[action] += update
        else:
            self.weights_2[action] += update

    
