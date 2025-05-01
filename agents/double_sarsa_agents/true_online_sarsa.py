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


class TrueOnlineSarsa(Base_Double_Sarsa_Agent):
    """
    Double Sarsa Agent that trains using an online sarsa approach. Adapted from "True Online Temporal-Difference Learning" Seijen et al. (2016)
    """
    def __init__(self, name, render=None, feature_type="reduced_ram"):
        super().__init__(name, render, feature_type)
        # Add just a single eligibility trace list + other hyperparameters
        self.e_traces = np.zeros((self.num_actions, self.feature_length))
        self.lam_val = 0.9
        self.Q_old = 0
        self.psi = None
        self.psi_prime = None
    
    def train(self, num_episodes=5000, bot_difficulty=0, render_mode=None):
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

        with open(f"logs/log_{self.name}.txt", "w") as log:
            log.write("Episode, Reward, TD Error, Epsilon, Loss (Squared TD Error)\n")
        # INPUT: α, λ, γ, θ[init]
        # alpha is self.alpha and gamma is self.gamma 
        # Loop (over episodes):
            for episode in range(num_episodes):
                # Perform extra save at episodes/2 checkpoint
                if episode % 500 == 0 and episode > 0:
                    save_linear_agent(self, f"{self.name}_{episode}")
                    # Plot rewards
                    # plot_rewards(self.rewards_history, 
                    #                 graph_name=f"(Episodes 0-{episode})",
                    #                 save_path=f"Learning Curves/{self.name}_png")
                    print(f"Agent saved at episode {episode}")

                # obtain initial state S
                state, _ = self.env.reset()
                state = self.feature_extraction(state)

                # select action A based on state S (epsilon-greedy)
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
                    
                    delta = np.clip(delta, -10.0, 10.0) 

                    
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

                squared_delta_error = delta ** 2
                entry = f"({episode}, {total_reward}, {delta}, {self.epsilon}, {squared_delta_error})\n"
                log.write(entry)
                
                if episode % 10 == 0:
                    avg_reward = np.mean(self.rewards_history[-10:])
                    print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        self.weights_2 = self.weights_1.copy() # Have only been updating one set of weights so get them to be the same 

        return self.rewards_history