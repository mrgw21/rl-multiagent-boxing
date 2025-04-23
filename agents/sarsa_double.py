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
from utils.agent_utils import load_agent, test_agent, save_agent, plot_rewards
from utils.lsh_script import LSH
from utils.PER import store_experience, prioritised_sample, update_priority_order


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
            self.feature_length = 10
        elif self.feature_type == "full_ram":
            self.feature_length = 524800  # Length of the features
        elif self.feature_type == "lsh":
            self.lsh = LSH()
            self.feature_length = self.lsh.num_rand_bit_vecs * self.lsh.hash_table_size
        else:
            raise ValueError("Feature Type not recognised")
        
        # Define general hyperparameters
        self.action_list = [i for i in range(18)]
        self.num_actions = len(self.action_list)
        # Rather than use zeros for weights, start with a random initialisation
        self.weights_1 = np.random.uniform(-0.01, 0.01, (self.num_actions, self.feature_length))
        self.weights_2 = np.random.uniform(-0.01, 0.01, (self.num_actions, self.feature_length))
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

        # Hyperparameters for eligibility traces using the cache method from Daley & Amato (2020) - https://arxiv.org/pdf/1810.09967
        self.e_traces_1 = np.zeros((self.num_actions, self.feature_length))
        self.e_traces_2 = np.zeros((self.num_actions, self.feature_length))
        self.lam_val = 0.9
        self.Q_old = 0
        self.psi = None
        self.psi_prime = None
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
        else:
            raise ValueError("Feature Type not recognised") # Won't actually be called 


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
        if update_weights is self.weights_1:
            self.weights_1[action] += self.alpha * td_error * state
        else:
            self.weights_2[action] += self.alpha * td_error * state


    
    def train_double_sarsa_lambda(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double SARSA(λ) approach with eligibility traces + prioritised experience replay.
        """
        # Ensure render mode is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"

        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Reset environment and eligibility traces
            state, _ = self.env.reset()
            state = self.feature_extraction(state)
            action = self.policy(state, online=True)  # Use online=True for initial action selection

            # Reset traces for both networks
            self.e_traces_1.fill(0.0)
            self.e_traces_2.fill(0.0)

            total_reward = 0
            finished = False
            steps_in_episode = 0

            # Halfway save/plot logic
            if episode > 0 and episode == num_episodes // 2:
                save_agent(self, f"{self.name}_lambda_per_halfway")
                plot_rewards(self.rewards_history,
                            graph_name=f"{self.name} Lambda PER (Episodes 0-{episode})",
                            save_path=f"{self.name}_lambda_per_halfway_learning_curve.png")
                print(f"Halfway results saved at episode {episode}")

            while not finished:
                # Environment step
                next_state_raw, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.feature_extraction(next_state_raw)
                finished = terminated or truncated
                next_action = self.policy(next_state, online=True)  # Choose next action (on-policy for SARSA)

                self.steps += 1
                steps_in_episode += 1
                total_reward += reward

                # --- On-Policy SARSA(λ) Update ---
                # Randomly choose which network to update and which to use for target
                if np.random.rand() < 0.5:
                    update_weights = self.weights_1
                    target_weights = self.target_weights_2  # Use target network for stability
                    e_traces = self.e_traces_1
                else:
                    update_weights = self.weights_2
                    target_weights = self.target_weights_1  # Use target network for stability
                    e_traces = self.e_traces_2

                # Calculate Q-values needed for TD error
                current_q_val = self.value(state, action, update_weights)
                next_q_val = 0.0
                if not finished:
                    # Get Q(s', a') from the *target* network corresponding to the *other* online network
                    next_q_val = self.value(next_state, next_action, target_weights)

                # Calculate TD error (delta)
                delta = reward + self.gamma * next_q_val - current_q_val

                # Update eligibility traces (Replacing traces for linear FA)
                # Decay existing traces
                e_traces *= self.gamma * self.lam_val
                # Add current state-action feature vector to the trace for the taken action
                # Ensure state is flat
                if state.ndim > 1:
                    state = state.flatten()
                if len(state) == self.feature_length:
                    e_traces[action] += state
                else:  # Handle mismatch if it occurs
                    print(f"Warning: State feature length {len(state)} != expected {self.feature_length} during trace update. Skipping trace increment.")

                # Update weights using traces and TD error
                update_weights += self.alpha * delta * e_traces  # Element-wise multiplication for linear FA update

                # --- Store Experience for PER ---
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(delta)  # Use magnitude of on-policy TD error for initial priority
                store_experience(self, experience, priority)

                # --- Prioritized Experience Replay (Off-policy TD(0) updates) ---
                if len(self.replay_buffer) > self.batch_size:
                    # Sample batch from replay buffer using priorities
                    sample_batch, sample_indices, importance_weights = prioritised_sample(self)
                    td_errors_for_update = []  # Store TD errors from replay batch

                    # Perform updates for each sampled experience
                    for i, sampled_experience in enumerate(sample_batch):
                        s_state, s_action, s_reward, s_next_state, s_next_action, s_finished = sampled_experience

                        # Randomly choose which network to update and which to use for target (Double learning for replay)
                        if np.random.rand() < 0.5:
                            s_update_weights = self.weights_1
                            s_target_weights = self.target_weights_2  # Use target network
                        else:
                            s_update_weights = self.weights_2
                            s_target_weights = self.target_weights_1  # Use target network

                        # Calculate target Q value for the sampled transition
                        s_current_q = self.value(s_state, s_action, s_update_weights)
                        s_next_q = 0.0
                        if not s_finished:
                            # Use Q(s', a') from the target network
                            s_next_q = self.value(s_next_state, s_next_action, s_target_weights)

                        # Calculate TD error for the sampled transition (TD(0))
                        s_td_error = s_reward + self.gamma * s_next_q - s_current_q
                        s_td_error = np.clip(s_td_error, -1.0, 1.0) 
                        td_errors_for_update.append(abs(s_td_error))  # Store error magnitude for priority update

                        # Update weights using TD error, importance sampling weight, and state features
                        # Ensure s_state is flat and has correct length
                        if s_state.ndim > 1:
                            s_state = s_state.flatten()
                        if len(s_state) == self.feature_length:
                            update_step = self.alpha * s_td_error * importance_weights[i] * s_state
                            s_update_weights[s_action] += update_step
                        else:
                            print(f"Warning: Sampled state feature length {len(s_state)} != expected {self.feature_length}. Skipping replay update.")

                    # Update priorities of the sampled experiences in the buffer
                    update_priority_order(self, sample_indices, td_errors_for_update)

                # Update target networks periodically
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()

                # Transition to next state
                state = next_state
                action = next_action

            # --- End of Episode ---
            # Decay epsilon and potentially alpha
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            # self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)  # Optional alpha decay

            # Anneal beta for importance sampling
            self.exp_beta = min(1.0, self.exp_beta + self.exp_beta_increment)

            self.rewards_history.append(total_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])  # Use rolling average (e.g., last 100)
                print(f"Episode: {episode}, Total Steps: {self.steps}, Ep Reward: {total_reward}, Avg Reward (100): {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}, Beta: {self.exp_beta:.4f}")

        # --- End of Training ---
        self.env.close()  # Clean up environment
        print("Training finished.")
        # Save final agent and plot rewards
        save_agent(self, f"{self.name}_lambda_per_final")
        plot_rewards(self.rewards_history,
                    graph_name=f"{self.name} Lambda PER (Final)",
                    save_path=f"{self.name}_lambda_per_final_learning_curve.png")

        return self.rewards_history
    
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
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway.pkl")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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
                    self.replay_buffer.pop()

                # Randomly choose betweening updating the first or second set of weights
                if np.random.rand() < 0.5:
                    target_weights = self.target_weights_1
                    update_weights = self.weights_2
                else:
                    target_weights = self.target_weights_2
                    update_weights = self.weights_1
                
                next_q_val = self.value(next_state, next_action, target_weights)
                current_q_val = self.value(state, action, update_weights)
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                td_error = np.clip(td_error, -1.0, 1.0) 
                
                self.update_weights(action, state, td_error, update_weights) # Perform update directly to agent's weights
                # Check if target weights need to be updated
                self.steps+= 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()
                
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
                        s_td_error = np.clip(s_td_error, -1.0, 1.0) 
                        self.update_weights(s_action, s_state, s_td_error, s_update_weights) # Perform update directly to agent's weights
                
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
    
    def train_double_sarsa_no_exp_elig(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach with random experience sampling. Using Hasselt (2010). 
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway.pkl")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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
                    target_weights = self.target_weights_1
                    update_weights = self.weights_2
                    e_traces = self.e_traces_1
                else:
                    target_weights = self.target_weights_2
                    update_weights = self.weights_1
                    e_traces = self.e_traces_2
                
                next_q_val = self.value(next_state, next_action, target_weights)
                current_q_val = self.value(state, action, update_weights)
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                td_error = np.clip(td_error, -1.0, 1.0) 


                delta = reward + self.gamma * next_q_val - current_q_val

                # Update eligibility traces (Replacing traces for linear FA)
                # Decay existing traces
                e_traces *= self.gamma * self.lam_val
                
                e_traces[action] += state
                update_weights += self.alpha * delta * e_traces  # Element-wise multiplication for linear FA update

                # Check if target weights need to be updated
                self.steps+= 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()
                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

            # Anneal beta for importance sampling
            self.exp_beta = min(1.0, self.exp_beta + self.exp_beta_increment)

            self.rewards_history.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        # Save weights
        # np.save(f"{self.name}_weights1.npy", self.weights_1)
        # np.save(f"{self.name}_weights2.npy", self.weights_2)
        return self.rewards_history

    def train_double_sarsa_no_exp_no_elig(self, num_episodes=5000, bot_difficulty=0, render_mode=None, normalise_features=True):
        """
        Trains the agent using a double sarsa approach with random experience sampling. Using Hasselt (2010). 
        """
        # Ensure render more is rgb_array if using lsh
        if self.feature_type == "lsh":
            render_mode = "rgb_array"
            obs_type = "rgb"
        else:
            obs_type = "ram"
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway.pkl")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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
                    target_weights = self.target_weights_1
                    update_weights = self.weights_2
                else:
                    target_weights = self.target_weights_2
                    update_weights = self.weights_1
                
                next_q_val = self.value(next_state, next_action, target_weights)
                current_q_val = self.value(state, action, update_weights)
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                td_error = np.clip(td_error, -1.0, 1.0) 
                
                self.update_weights(action, state, td_error, update_weights) # Perform update directly to agent's weights
                # Check if target weights need to be updated
                self.steps+= 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()
                
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
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway.pkl")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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
                    target_weights = self.target_weights_1
                    update_weights = self.weights_2
                else:
                    target_weights = self.target_weights_2
                    update_weights = self.weights_1
                
                next_q_val = self.value(next_state, next_action, target_weights)
                current_q_val = self.value(state, action, update_weights)
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                td_error = np.clip(td_error, -1.0, 1.0) 
                self.update_weights(action, state, td_error, update_weights) # Perform update directly to agent's weights
                
                # Store experience + priority 
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(td_error) # Magnitude of TD error
                store_experience(self,experience, priority, cache=False)

                # Check if target weights need to be updated
                self.steps += 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()
        
                # Perform experience replay if there are enough samples
                if len(self.replay_buffer) > self.batch_size:

                    # Extract a set of samples, their indices and importance sampling weights using prioritised_sample(self)
                    samples, sample_indices, importance_weights = prioritised_sample(self)
                    td_errors = []
                    
                    # For each sample in the batch, run double learning loop again
                    for idx, sampled_experience in enumerate(samples):
                        s_state, s_action, s_reward, s_next_state, s_next_action, s_finished = sampled_experience
                        
                        # Choose weights randomly for each sample
                        if np.random.rand() < 0.5:
                            s_target_weights = self.weights_1
                            s_update_weights = self.weights_2
                        else:
                            s_target_weights = self.weights_2
                            s_update_weights = self.weights_1
                        
                        s_next_q_val = self.value(s_next_state, s_next_action, s_target_weights)
                        s_current_q_val = self.value(s_state, s_action, s_update_weights)

                        # Calculate difference between target and estimate q values again
                        if s_finished:
                            s_td_error = s_reward - s_current_q_val
                        else:
                            s_td_error = s_reward + self.gamma * s_next_q_val - s_current_q_val
                        
                        s_td_error = np.clip(s_td_error, -1.0, 1.0) 
                        # Update the selected set of weights with importance sampling
                        if s_update_weights is self.weights_1:
                            self.weights_1 += self.alpha * s_td_error * importance_weights[idx] * s_state
                        else:
                            self.weights_2 += self.alpha * s_td_error * importance_weights[idx] * s_state

                        td_errors.append(abs(s_td_error))
                    
                    # Update priorities once the new sample has been added
                    update_priority_order(self, sample_indices, td_errors)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            # Adjust exp_beta for importance sampling 
            beta_increment = (1.0 - self.initial_beta) / num_episodes
            self.exp_beta = min(1.0, self.exp_beta + beta_increment)

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
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway.pkl")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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
                    target_weights = self.target_weights_1
                    update_weights = self.weights_2
                else:
                    target_weights = self.target_weights_2
                    update_weights = self.weights_1
                
                next_q_val = self.value(next_state, next_action, target_weights)
                current_q_val = self.value(state, action, update_weights)
                
                # Calculate TD error - difference between target and current value estimate
                if finished:
                    td_error = reward - current_q_val
                else:
                    td_error = reward + self.gamma * next_q_val - current_q_val
                
                td_error = np.clip(td_error, -1.0, 1.0) 
                # self.update_weights(action, state, td_error, update_weights) # Perform update directly to agent's weights
                
                # Store experience + priority 
                experience = (state, action, reward, next_state, next_action, finished)
                priority = abs(td_error) # Magnitude of TD error
                store_experience(self,experience, priority, cache=True)

                # Check if target weights need to be updated
                self.steps+= 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()

                # Perform immediate TD update if random value is greater than probability or cache is too small
                if np.random.rand() < self.cache_update_probability and len(self.cache) >= self.batch_size:
                    # Sample batch_size samples from the cache - unique values only
                    sample_size = min(self.batch_size, len(self.cache))
                    samples = random.sample(self.cache, sample_size)

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
                        s_td_error = np.clip(s_td_error, -1.0, 1.0) 

                        self.update_weights(s_action,  s_state, s_td_error, s_update_weights) # Perform update directly to agent's weights
                else:
                    # Update agent's weights, not local weights
                    self.update_weights(action, state, td_error, update_weights) # Perform update directly to agent's weights
                                
                state = next_state
                action = next_action
                total_reward += reward
            
            # Epsilon + Alpha decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        
            # Adjsut exp_beta for importance sampling 
            beta_increment = (1.0 - self.initial_beta) / num_episodes
            self.exp_beta = min(1.0, self.exp_beta + beta_increment)

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
            
        self.env = gym.make("ALE/Boxing-ram-v5", obs_type=obs_type, render_mode=render_mode) # difficulty=bot_difficulty)
        print(f"Training with difficulty: {bot_difficulty}")

        # Add just a single eligibility trace list
        self.e_traces = np.zeros((self.num_actions, self.feature_length))
        # INPUT: α, λ, γ, θ[init]
        # alpha is self.alpha and gamma is self.gamma 
        # Loop (over episodes):
        for episode in range(num_episodes):
            # Perform extra save at episodes/2 checkpoint
            if episode == num_episodes // 2:
                save_agent(self, f"{self.name}_halfway")
                # Plot rewards
                plot_rewards(self.rewards_history, 
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

                self.e_traces[action] += self.psi
                self.e_traces[action] -= self.alpha * self.gamma * self.lam_val * np.dot(self.e_traces[action], self.psi) * self.psi

                # θ ← θ + α(δ + Q − Qold) e − α(Q − Qold)ψ
                # Only update for the relevant action again
                self.weights_1[action] += self.alpha * (delta + Q - self.Q_old) * self.e_traces[action] - \
                                          self.alpha * (Q - self.Q_old) * self.psi

                # Qold ← Q′
                self.Q_old = Q_prime
                
                # ψ ← ψ′ ; A ← A′
                self.psi = self.psi_prime
                action = next_action

                total_reward += reward

                # Check if target weights need to be updated
                self.steps+= 1
                if self.steps % self.target_update_freq == 0:
                    self.update_target_networks()
            
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
    agent = Double_SARSA_Agent("Double_SARSA_Boxing_22_04_train_prioritised_cache_full", render=None, feature_type="full_ram")
    print(agent.name)
    rewards = agent.train_double_sarsa_with_cache(num_episodes=5000) # Default difficulty

    save_agent(agent, agent.name)
    loaded_agent = load_agent(agent.name)
    test_agent(loaded_agent, render_mode=None)

    plot_rewards(rewards, "Final Agent Rewards", save_path=f"Learning Curves/{agent.name} Final Curve")

