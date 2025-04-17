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
# if using gymnasium
import shimmy
import time

import gymnasium as gym
import random
from agent_utils import *



class Double_SARSA_Agent:
    """
    Double SARSA with prioritised experience replay for Boxing
    """
    def __init__(self, name, render=None):
        # Define general hyperparameters
        self.feature_length = 7  # Only 7 because using reduced feature space
        self.num_actions = 18
        self.weights_1 = np.zeros((self.num_actions, self.feature_length))
        self.weights_2 = np.zeros((self.num_actions, self.feature_length))
        self.alpha = 0.1
        self.epsilon = 0.1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.alpha_decay = 0.9999
        self.alpha_min = 0.01
        self.gamma = 0.9
        
        # Prioritised experience replay using Schaul et al. (2015) - Prioritized Experience Replay).
        self.max_capacity = 10000
        self.replay_buffer = []
        self.priorities = []
        self.batch_size = 32
        # "how much prioritisation is used" - when exp_alpha is 0, all samples have the same probability of being sampled
        self.exp_alpha = 0.6
        # Importance sampling exponent - prevents bias of sampling from only high priority samples
        self.exp_beta = 0.4  
        
        # Create env using ram observation type
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render)
        
        # Store agent name and rewards
        self.name = name
        self.rewards_history = []
        
    def feature_extraction(self, ram_data, normalise):
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
        # player_score = ram_data[18] 
        # opponent_score = ram_data[19]
        # timer = ram_data[59]
        
        features = np.array([player_x, player_y, opponent_x, opponent_y, 
                            dx, dy, distance])
        
        # Normalise features - could try removing/tweaking this
        if normalise:
            features = features / 255.0
        return features
    
    def value(self, feature_space, action, weights):
        """
        Return value of action in a given feature space - simple linear approximation using dot product
        """
        return np.dot(feature_space, weights[action])
    
    def policy(self, state):
        """
        Epsilon-greedy policy using average of the two weights - needed for double learning
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample() # Random action
        else:
            # Use average value from both weights to get action with highest q value
            q_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                q_values[action] = (self.value(state, action, self.weights_1) + 
                                    self.value(state, action, self.weights_2)) / 2 
            return np.argmax(q_values)
    
    def store_experience(self, experience, priority):
        """
        Stores an experience + its priority in the relevant lists
        """
        # If we are over capacity, remove the lowest priority experience + priority (last one)
        if len(self.replay_buffer) >= self.max_capacity:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)
        # Add new experience and priority
        self.replay_buffer.append(experience)
        self.priorities.append(priority)
    
    def prioritised_sample(self):
        """
        Sample a set of experiences based on their value/priority - determined using td_error
        """
        if not self.replay_buffer:
            return []
        
        # Create a copy array for the priorities 
        priority_arr = np.array(self.priorities, dtype=np.float64)

        # Additional check to make sure no values are nan in the prioity list or the total sum = 0
        if np.sum(priority_arr) == 0 or np.any(np.isnan(priority_arr)):
            priority_arr = np.ones_like(priority_arr) # Sets all priorities to 1 so probability of selecting experience is equal

        # Create probability array using alpha priority value -  determines probability of choosing action from the priority values (Schaul et al., 2015)
        probs = priority_arr ** self.exp_alpha
        probs /= np.sum(probs)  # Divide the array by its own sum to ensure it sums up to 1 (needed to for a prbability distribution)
        
        # An extra check to prevent nan probabilities
        if np.any(np.isnan(probs)):
            probs = np.ones(len(priority_arr)) / len(priority_arr) # Make probabilities all the same in the array 

        # Randomly choose the indices of the experiences using these probabilities - replace prevents sampling the same experience twice
        idx_vals = np.random.choice(len(self.replay_buffer), min(self.batch_size, 
                                len(self.replay_buffer)), p=probs, replace=False)
        
        # Importance sampling weights to stop the agent only choosing high priority values - from Schaul et al. (2015)
        weights = (len(self.replay_buffer) * probs[idx_vals]) ** (-self.exp_beta)
        weights = weights / np.max(weights)  # Same normalisations as before
        
        # Use calculated indices to extract experiences from self.replay_buffer
        sample_batch = [self.replay_buffer[i] for i in idx_vals]
        return sample_batch, idx_vals, weights
    
    def update_priority_order(self, indices, errors):
        """
        Update the priority list with the latest TD errors
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6  # Add a small constant to the absolute TD error to make it more stable
    
    def train(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach. Adapted from Schaul et al. (2015) and Hasselt (2010).
        """
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render_mode, difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.feature_extraction(state, normalise=normalise_features) # only get the features we need
            
            action = self.policy(state)
            total_reward = 0
            
            finished = False
            episode_step = 0
            
            while not finished:
                episode_step += 1
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state, normalise=normalise_features) # only get the features we need
                finished = terminated or truncated # Can be either in the Atari envs
                
                next_action = self.policy(next_state)
                
                # Randomly choose betweening updating the first or second set of weights
                if np.random.rand() < 0.5:
                    weights = self.weights_1
                    update_weights = self.weights_2
                else:
                    weights = self.weights_2
                    update_weights = self.weights_1
                
                # To perform double learning:
                # - use one set of weights to estimate value of next state-action
                next_q_val = self.value(next_state, next_action, weights)
                # - use other set of weights to estimate current state-action
                current_q_val = self.value(state, action, update_weights)
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                # Store experience + priority 
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(td_error) # Magnitude of TD error
                self.store_experience(experience, priority)
                
                # Perform experience replay if there are enough samples
                if len(self.replay_buffer) > self.batch_size:

                    # Extract a set of samples, their indices and importance sampling weights using prioritised_sample()
                    samples, sample_indices, importance_weights = self.prioritised_sample()
                    td_errors = []
                    
                    # For each sample in the batch, run double learning loop again
                    for idx, sampled_experience in enumerate(samples):
                        s_state, s_action, s_reward, s_next_state, s_next_action, s_finished = sampled_experience
                        
                        # Choose weights randomly for each sample
                        if np.random.rand() < 0.5:
                            s_weights = self.weights_1
                            s_update_weights = self.weights_2
                        else:
                            s_weights = self.weights_2
                            s_update_weights = self.weights_1
                        
                        if s_finished:
                            s_target_q_val = s_reward
                        else:
                            s_target_q_val = s_reward + self.gamma * self.value(s_next_state, s_next_action, s_weights)
                        
                        # Calculate difference between target and estimate q values again
                        s_current_q_val = self.value(s_state, s_action, s_update_weights)
                        s_td_error = s_target_q_val - s_current_q_val
                        
                        # Update the selected set of weights with importance sampling
                        s_update_weights[s_action] += self.alpha * s_td_error * importance_weights[idx] * s_state
                        td_errors.append(s_td_error)
                    
                    # Update priorities once the new sample has been added
                    self.update_priority_order(sample_indices, td_errors)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            # Increase exp_beta for importance sampling 
            exp_beta_increase = 1.0 / num_episodes
            self.exp_beta = min(1.0, self.exp_beta + exp_beta_increase) # Stop it from going below 1

            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        # Save weights
        np.save(f"{self.name}_weights1.npy", self.weights_1)
        np.save(f"{self.name}_weights2.npy", self.weights_2)
        return self.rewards_history


# Only run training if we are running this script - needed when importing module from another script
if __name__ == "__main__":
    agent = Double_SARSA_Agent("Double_SARSA_Boxing_17_04_no_norm")

    # 
    rewards = agent.train(num_episodes=10000, bot_difficulty=3, normalise_features=False)

    save_agent(agent, "saved_agents/double_sarsa_17_04_no_norm.pkl")
    loaded_agent = load_agent("saved_agents/double_sarsa_17_04_no_norm.pkl")
    test_agent(loaded_agent)

    def moving_average(data, window_size=50):
        # Just performs moving average calculation to plot on graph + help with interpretability of graph
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(f'Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.plot(moving_average(rewards), label='Smoothed')
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.savefig(f"Double_Sarsa_learning_curve_best2_no_norm.png")
    plt.show()


