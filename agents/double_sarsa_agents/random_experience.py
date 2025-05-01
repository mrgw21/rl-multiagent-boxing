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
from .sarsa_double import Base_Double_Sarsa_Agent


class DoubleSarsaRandomExperience(Base_Double_Sarsa_Agent):
    """
    Double Sarsa Agent that uses randomly sampled experiences from an experience replay.
    """
    def __init__(self, name, render=None, feature_type="reduced_ram",):
        super().__init__(name, render, feature_type)
        
    def train(self, num_episodes=5000, bot_difficulty=0, render_mode=None):
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

        with open(f"logs/log_{self.name}.txt", "w") as log:
            log.write("Episode, Reward, TD Error, Epsilon, Loss (Squared TD Error)\n")
            
            for episode in range(num_episodes):
                # Perform extra save at episodes/2 checkpoint
                if episode % 500 == 0 and episode > 0:
                    save_linear_agent(self, f"{self.name}_{episode}")
                    # Plot rewards
                    # plot_rewards(self.rewards_history, 
                    #                 graph_name=f"(Episodes 0-{episode})",
                    #                 save_path=f"Learning Curves/{self.name}_png")
                    print(f"Agent saved at episode {episode}")

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
                    self.replay_buffer.append(experience)

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
                    
                    td_error = np.clip(td_error, -10.0, 10.0) 

                    
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
                            s_td_error = np.clip(s_td_error, -10.0, 10.0) 
                            self.update_weights(s_action, s_state, s_td_error, s_update_weights) # Perform update directly to agent's weights
                    
                    state = next_state
                    action = next_action
                    total_reward += reward
                
                # Epsilon + Alpha decay
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

                self.rewards_history.append(total_reward)

                squared_td_error = td_error ** 2
                entry = f"({episode}, {total_reward}, {td_error}, {self.epsilon}, {squared_td_error})\n"
                log.write(entry)
                
                if episode % 10 == 0:
                    avg_reward = np.mean(self.rewards_history[-10:])
                    print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                
        return self.rewards_history