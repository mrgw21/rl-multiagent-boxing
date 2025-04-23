
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
from .sarsa_double import Base_Double_Sarsa_Agent


class DoubleSarsaNoExperience(Base_Double_Sarsa_Agent):
    """
    Double Sarsa Agent that uses a double sarsa approach with no experience replay - Double learning adapted from using Hasselt (2010). 
    """
    def __init__(self, name, render=None, feature_type="reduced_ram"):
        super().__init__(name, render, feature_type)
        self.e_traces_1 = np.zeros((self.num_actions, self.feature_length))
        self.e_traces_2 = np.zeros((self.num_actions, self.feature_length))
        self.lam_val = 0.9


    def train(self, num_episodes=5000, bot_difficulty=0, render_mode=None):

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
                save_agent(self, f"{self.name}_halfway")
                # Plot rewards
                plot_rewards(self.rewards_history, 
                                 graph_name=f"(Episodes 0-{episode})",
                                 save_path=f"Learning Curves/{self.name}_halfway_learning_curve.png")
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
                
        return self.rewards_history