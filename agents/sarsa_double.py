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
from lsh_script import LSH


class Double_SARSA_Agent:
    """
    Double SARSA with prioritised experience replay for Boxing
    """
    def __init__(self, name, render=None, feature_type="reduced_ram"):
        """
        feature_type can be either "reduced_ram", "full_ram" or "lsh"
        """
        # Determine feature type
        self.feature_type = feature_type
        if self.feature_type == "reduced_ram":
            self.feature_length = 7
        elif self.feature_type == "full_ram":
            self.feature_length = 524800  # Length of the features
        elif self.feature_type == "lsh":
            self.lsh = LSH()
            self.feature_length = self.lsh.num_rand_bit_vecs * self.lsh.hash_table_size
        else:
            raise ValueError("Feature Type not recognised")
        
        # Define general hyperparameters
        self.num_actions = 18
        # Rather than use zeros for weights, start with a random initialisation
        self.weights_1 = np.random.uniform(-0.01, 0.01, (self.num_actions, self.feature_length))
        self.weights_2 = np.random.uniform(-0.01, 0.01, (self.num_actions, self.feature_length))
        self.alpha = 0.1
        self.epsilon = 0.2
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

        # Hyperparameters for eligibility traces using the cache method from Daley & Amato (2020) - https://arxiv.org/pdf/1810.09967
        self.e_traces = np.zeros((self.num_actions, self.feature_length))
        self.Q_old = 0
        self.psi = None
        self.psi_prime = None
        self.lam_val = 0.9
        self.cache_size = 80000
        self.block_size = 100
        self.refresh_frequency = 10000
        self.cache = []
        self.steps_since_last_update = 0
        self.cache_update_probability = 0.7 # Probability of using cache to perform update vs. immediate TD val

        self.normalise_features = True

        # Create env using ram observation type
        if self.feature_type == "lsh":
            self.env = gym.make("ALE/Boxing-ram-v5", obs_type="rgb", render_mode="rgb_array")
        else:
            self.env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render)
        
        # Store agent name and rewards
        self.name = name
        self.rewards_history = []

    
    def feature_extraction(self, observations):
        # Determine feature type
        if self.feature_type == "reduced_ram":
            return self.reduced_feature_extraction_ram(observations)
        elif self.feature_type == "full_ram":
            return self.feature_extraction_ram(observations)
        elif self.feature_type == "lsh":
            screen = self.env.render()
            return self.lsh.feature_extraction_lsh(screen)
        else:
            raise ValueError("Feature Type not recognised") # Won't actually be called ever


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
        # player_score = ram_data[18] 
        # opponent_score = ram_data[19]
        # timer = ram_data[59]
        
        features = np.array([player_x, player_y, opponent_x, opponent_y, 
                            dx, dy, distance])
        
        # Normalise features - could try removing/tweaking this
        if self.normalise_features:
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
    
    def store_experience(self, experience, priority, cache=True):
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

        if cache:
            # Refresh cache every self.refresh_frequency steps
            self.steps_since_last_update += 1
            if self.steps_since_last_update >= self.refresh_frequency:
                self.update_cache() # build_cache function from Daley & Amato
                self.steps_since_last_update = 0
        
    def update_cache(self):
        """ 
        Implementatino of the build_cache function from Daley & Amato - refreshes the cache
        """
        self.cache = [] # Initialize empty list C
        # for 1, 2, . . . , S/B do
        for i in range(self.cache_size // self.block_size):
            # Sample block (sk, ak, rk, sk+1), . . . , (sk+B−1, ak+B−1, rk+B−1, sk+B ) from D
            if len(self.replay_buffer) <= self.block_size:
                sample_block = self.replay_buffer.copy()
            else:
                sk = np.random.randint(0, len(self.replay_buffer) - self.block_size)
                sample_block = self.replay_buffer[sk:sk + self.block_size]
            
            sample_block = self.replay_buffer[sk : sk+self.block_size]

            # Rλ ← max(a′ ∈ A) Q(sk+B,a′;θ)
            lam_returns = []
            next_return = 0

            #for i ∈ {k + B − 1, k + B − 2, . . . , k} do
            for sample in reversed(sample_block):
                state, action, reward, next_state, next_action, finished = sample

                # Rλ ← ri
                if finished:
                    lam_return = reward
                # Rλ ← ri + γ[λRλ + (1 − λ) maxa′ ∈A Q(ˆsi+1, a′; θ)]
                else:
                    q_val_1 = self.value(next_state, next_action, self.weights_1) 
                    q_val_2 = self.value(next_state, next_action, self.weights_2) 
                    q_estimate = (q_val_1 + q_val_2) / 2
                    lam_return = reward + self.gamma * (self.lam_val * next_return + (1-self.lam_val) * q_estimate)
                #Append tuple (ˆsi, ai, Rλ) to C
                lam_returns.append((state, action, lam_return))
                next_return = lam_return

            self.cache.extend(reversed(lam_returns))

    
    def prioritised_sample(self):
        """
        Sample a set of experiences based on their value/priority - determined using td_error
        """
        if not self.replay_buffer:
            return [], [], []
        
        # Create a copy array for the priorities 
        priority_arr = np.array(self.priorities, dtype=np.float64)

        # Additional check to make sure no values are nan in the prioity list or the total sum = 0
        if np.sum(priority_arr) == 0 or np.any(np.isnan(priority_arr)):
            priority_arr = np.ones_like(priority_arr) # Sets all priorities to 1 so probability of selecting experience is equal

        # Create probability array using alpha priority value -  determines probability of choosing action from the priority values (Schaul et al., 2015)
        probs = priority_arr ** self.exp_alpha
        probs /= np.sum(probs)  # Divide the array by its own sum to ensure it sums up to 1 (needed to for a prbability distribution)
        
        # To ensure sample size doesn't exceed number of non-zero probabilities
        num_non_zeros = np.count_nonzero(probs)
        # Calculate possible sample size by taking min of possible values
        sample_size = min(self.batch_size, len(self.replay_buffer), num_non_zeros)

        # If sample size is 0, use uniform sampling again
        if sample_size == 0:
            probs = np.ones(len(priority_arr)) / len(priority_arr)
            sample_size = min(self.batch_size, len(self.replay_buffer))

        # An extra check to prevent nan probabilities
        if np.any(np.isnan(probs)):
            probs = np.ones(len(priority_arr)) / len(priority_arr) # Make probabilities all the same in the array 
            sample_size = min(self.batch_size, len(self.replay_buffer))

        # Can reuse samples if sample size is greater than the number of non zero samples
        if sample_size > num_non_zeros:
            reuse_samples = True
        else:
            reuse_samples = False
        # Randomly choose the indices of the experiences using these probabilities - replace prevents sampling the same experience twice
        idx_vals = np.random.choice(len(self.replay_buffer), min(self.batch_size, 
                                len(self.replay_buffer)), p=probs, replace=reuse_samples)
        
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
    
    def update_weights(self, action, state, td_error, update_weights):
        """Performs update directly to agent's weight vectors rather than to local weights"""
        if update_weights is self.weights_1:
            self.weights_1[action] += self.alpha * td_error * state
        else:
            self.weights_2[action] += self.alpha * td_error * state
    
    def plot_rewards(self, rewards, graph_name, save_path=None):
        """
        Function to plot and save learning curves
        """
        def moving_average(data, window_size=500):
            # Just performs moving average calculation to help with interpretability 
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(rewards), linewidth=2, color='red', label='Smoothed')
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
        plt.title(f'Learning Curve {graph_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def train_double_sarsa_rand_exp(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach with random experience sampling. Using Hasselt (2010). 
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode, difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"saved_agents/{self.name}_halfway.pkl")
                # Plot rewards
                self.plot_rewards(self.rewards_history, 
                                 graph_name=f"(Episodes 0-{episode})",
                                 save_path=f"{self.name}_halfway_learning_curve.png")
                print(f"Halfway results saved at episode {episode}")

            state, _ = self.env.reset()
            state = self.feature_extraction(state) 
            
            action = self.policy(state)
            total_reward = 0
            finished = False
            
            while not finished:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state) # only get the features we need
                finished = terminated or truncated # Can be either in the Atari envs
                next_action = self.policy(next_state)

                # Store experience 
                experience = (state, action, reward, next_state, next_action, finished)
                # Remove an experience from buffer if at capacity
                if len(self.replay_buffer) >= self.max_capacity:
                    self.replay_buffer.pop(0)
                self.replay_buffer.append(experience)

                # Randomly choose betweening updating the first or second set of weights
                if np.random.rand() < 0.5:
                    weights = self.weights_1
                    update_weights = self.weights_2
                else:
                    weights = self.weights_2
                    update_weights = self.weights_1
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_target = reward
                else:
                    # - use one set of weights to estimate value of next state-action
                    next_q_val = self.value(next_state, next_action, weights)
                    td_target = reward + self.gamma * next_q_val 
                
                # - use other set of weights to estimate current state-action
                current_q_val = self.value(state, action, update_weights)
                td_error = td_target - current_q_val
                self.update_weights(action, td_error, state, update_weights) # Perform update directly to agent's weights
                
                # Perform experience replay if there are enough samples
                if len(self.replay_buffer) > self.batch_size:

                    # get random batch of samples
                    sample_batch = random.sample(self.replay_buffer, self.batch_size)
                    
                    # For each sample in the batch, run double learning loop again
                    for sampled_experience in sample_batch:
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
                        self.update_weights(s_action, s_td_error, s_state, s_update_weights) # Perform update directly to agent's weights
                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        # Save weights
        # np.save(f"{self.name}_weights1.npy", self.weights_1)
        # np.save(f"{self.name}_weights2.npy", self.weights_2)
        return self.rewards_history
    
    def train_double_sarsa_no_cache(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach. Adapted from Schaul et al. (2015) and Hasselt (2010).
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode, difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"saved_agents/{self.name}_halfway.pkl")
                # Plot rewards
                self.plot_rewards(self.rewards_history, 
                                 graph_name=f"(Episodes 0-{episode})",
                                 save_path=f"{self.name}_halfway_learning_curve.png")
                print(f"Halfway results saved at episode {episode}")

            state, _ = self.env.reset()
            state = self.feature_extraction(state) 
            
            action = self.policy(state)
            total_reward = 0
            
            finished = False
            
            while not finished:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state) # only get the features we need
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
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                # Store experience + priority 
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(td_error) # Magnitude of TD error
                self.store_experience(experience, priority, cache=False)
                
                # Update weights 
                self.update_weights(action, td_error, state, update_weights) # Perform update directly to agent's weights
        
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
                        if s_update_weights is self.weights_1:
                            self.weights_1 += self.alpha * s_td_error * importance_weights[idx] * s_state
                        else:
                            self.weights_2 += self.alpha * s_td_error * importance_weights[idx] * s_state

                        td_errors.append(abs(s_td_error))
                    
                    # Update priorities once the new sample has been added
                    self.update_priority_order(sample_indices, td_errors)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            # Adjust exp_beta for importance sampling 
            self.exp_beta = min(1.0, self.exp_beta + (1.0 - self.exp_beta) * 0.01)

            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        # Save weights
        # np.save(f"{self.name}_weights1.npy", self.weights_1)
        # np.save(f"{self.name}_weights2.npy", self.weights_2)
        return self.rewards_history
    
    def train_double_sarsa_with_cache(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach with a cache for eligibility traces + prioritised experirence replay. Adapted from Daley and Amato (2015).
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode, difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"saved_agents/{self.name}_halfway.pkl")
                # Plot rewards
                self.plot_rewards(self.rewards_history, 
                                 graph_name=f"(Episodes 0-{episode})",
                                 save_path=f"{self.name}_halfway_learning_curve.png")
                print(f"Halfway results saved at episode {episode}")

            state, _ = self.env.reset()
            state = self.feature_extraction(state)
            action = self.policy(state)
            total_reward = 0
            finished = False
            
            while not finished:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state) # only get the features we need
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
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                # Store experience + priority 
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(td_error) # Magnitude of TD error
                self.store_experience(experience, priority, cache=True)
                
                # Perform immediate TD update if random value is greater than probability or cache is too small
                if np.random.rand() > self.cache_update_probability or len(self.cache) < self.batch_size:
                    # Update agent's weights, not local weights
                    self.update_weights(action, td_error, state, update_weights) # Perform update directly to agent's weights
                else:
                    # Sample batch_size samples from the cache - unique values only
                    samples = random.sample(self.cache, min(self.batch_size, len(self.cache)))

                    for sample in samples:
                        s_state, s_action, s_lam_return = sample

                        #Randomly choose which weights to update
                        if np.random.rand() < 0.5:
                            s_update_weights = self.weights_1
                        else:
                            s_update_weights = self.weights_2
                        
                        # Calculate error using lambda return
                        s_current_q_val = self.value(s_state, s_action, s_update_weights)
                        s_td_error = s_lam_return - s_current_q_val

                        self.update_weights(s_action, s_td_error, s_state, s_update_weights) # Perform update directly to agent's weights
                        
                                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            # Adjsut exp_beta for importance sampling 
            self.exp_beta = min(1.0, self.exp_beta + (1.0/num_episodes))

            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        # Save weights
        # np.save(f"{self.name}_weights1.npy", self.weights_1)
        # np.save(f"{self.name}_weights2.npy", self.weights_2)
        return self.rewards_history
    
    def train_true_online_sarsa(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using an online sarsa approach. Adapted from "True Online Temporal-Difference Learning" Seijen et al. (2016)
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode, difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        # INPUT: α, λ, γ, θ[init]
        # alpha is self.alpha and gamma is self.gamma 
        # Loop (over episodes):
        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"saved_agents/{self.name}_halfway.pkl")
                # Plot rewards
                self.plot_rewards(self.rewards_history, 
                                 graph_name=f"(Episodes 0-{episode})",
                                 save_path=f"{self.name}_halfway_learning_curve.png")
                print(f"Halfway results saved at episode {episode}")

            # obtain initial state S
            state, _ = self.env.reset()
            state = self.feature_extraction(state)

            # select action A based on state S (epislon-greedy)
            action = self.policy(state)

            # ψ ← features corresponding to S, A
            self.psi = state.copy()

            total_reward = 0
            episode_step = 0
            finished = False

            # While terminal state has not been reached, do:
            while not finished:
                episode_step += 1
                
                # take action A, observe next state S′ and reward R
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state) # only get the features we need
                finished = terminated or truncated # Can be either in the Atari envs
                
                # select action A′ based on state S′
                next_action = self.policy(next_state)

                # ψ′ ← features corresponding to S′, A′ (if S′ is terminal state, ψ′ ← 0)
                if finished:
                    self.psi_prime = np.zeros_like(state)
                else:
                    self.psi_prime = next_state.copy()

                # Q ← θ^T ψ
                Q = self.value(self.psi, action, self.weights_1)

                # Q' ← θ^T ψ' 
                # δ ← R + γ Q′ − Q
                # if finished, expected future reward (Q') will be 0 
                if finished:
                    Q_prime = 0
                else:
                    Q_prime = self.value(self.psi_prime, next_action, self.weights_1)
                
                if finished:
                    delta = reward - Q
                else:
                    delta  = reward + (self.gamma * Q_prime) - Q
                
                # e ← γλe + ψ - αγλ(e^T ψ)ψ
                self.e_traces *= self.gamma * self.lam_val
                for possible_action in range(self.num_actions):
                    # Update the eligibility traces for the current action - not all of them
                    if possible_action == action:
                        self.e_traces[possible_action] = self.gamma * self.lam_val * self.e_traces[possible_action] + self.psi \
                            - self.alpha * self.gamma * self.lam_val * np.dot(self.e_traces[possible_action], self.psi) * self.psi
                    else:
                        self.e_traces[possible_action] = self.gamma * self.lam_val * self.e_traces[possible_action]


                # θ ← θ + α(δ + Q − Qold) e − α(Q − Qold)ψ
                # Only update for the relevant action again
                self.weights_1 += self.alpha * (delta + Q - self.Q_old) * self.e_traces - self.alpha * (Q - self.Q_old) * np.array([self.psi if a == action else np.zeros_like(self.psi) for a in range(self.num_actions)])

                # Qold ← Q′
                self.Q_old = Q_prime
                
                # ψ ← ψ′ ; A ← A′
                self.psi = self.psi_prime
                action = next_action

                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            
            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        self.weights_2 = self.weights_1.copy()
        return self.rewards_history

        



# Only run training if we are running this script - needed when importing module from another script
if __name__ == "__main__":
    agent = Double_SARSA_Agent("Double_SARSA_Boxing_19_04_online_3", render=None, feature_type="reduced_ram")
    print(agent.name)
    rewards = agent.train_true_online_sarsa(num_episodes=5000, bot_difficulty=0) # Default difficulty

    save_agent(agent, f"saved_agents/{agent.name}.pkl")
    loaded_agent = load_agent(f"saved_agents/{agent.name}.pkl")
    test_agent(loaded_agent, render_mode=None)

    agent.plot_rewards(rewards, "Final Agent Rewards", f"{agent.name} Final Curve")

