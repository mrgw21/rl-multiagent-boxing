import numpy as np
import os
import cv2
from pettingzoo.atari import boxing_v2
from tile_coding import TileCoder
import matplotlib.pyplot as plt
import time
from collections import deque
import random 

class Standard_Agent:
    def __init__(self, tile_coder=None):
        self.feature_length = 522752 + 18
        self.weights = np.zeros(self.feature_length)
        self.alpha = 0.01  # Experiment with this
        self.epsilon = 0.15  # Start value
        self.gamma = 0.9
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.min_epsilon = 0.01

        self.env = boxing_v2.parallel_env(obs_type='ram', render_mode=None)
        self.ID = "first_0"
        self.env.reset()
        self.actions = list(range(self.env.action_space(self.ID).n))
        
        if tile_coder:
            self.tile_coder = tile_coder
            self.tile_feature_length = self.tile_coder.num_tile_grids * np.prod(self.tile_coder.bins)
            self.feature_length = self.tile_feature_length * len(self.actions)
            self.weights = np.zeros(self.feature_length)


        self.alpha = 0.1
        self.min_alpha = 0.01
        self.alpha_decay = 0.995

        # Experience replay
        self.experiences = deque(maxlen=5000)
        self.sample_size = 50

        # Initialise weights
        self.Policy = {}
        self.num_features = 4  # [px, py, ox, oy]
        self.num_actions = len(self.actions)

        # Flat linear weights: theta vectors are 1D of length (num_actions * num_features)
        self.theta1 = np.zeros(self.num_actions * self.num_features)
        self.theta2 = np.zeros(self.num_actions * self.num_features)


    def feature_extraction(self, observations):
        binary_observations = np.unpackbits(observations)
        i, j = np.triu_indices(len(binary_observations), k=1)
        pairwise_ands = binary_observations[i] & binary_observations[j]
        features = np.concatenate([binary_observations, pairwise_ands])
        return features


    # def learn_Sarsa(self, episodes=100):
    #     episode_rewards = []
    #     for episode in range(episodes):
    #         observations, _ = self.env.reset()
    #         ram = observations[self.ID]
    #         state_features = self.get_discrete_state(ram)
    #         action = self.policy(state_features)
    #         total_reward, episode_step = 0, 0

    #         while True:
    #             actions = {agent: action if agent == self.ID else np.random.choice(self.actions)
    #                        for agent in self.env.agents}
    #             observations, rewards, terminations, truncations, _ = self.env.step(actions)
    #             reward = rewards[self.ID]
    #             terminal = terminations[self.ID] or truncations[self.ID]

    #             ram_next = observations[self.ID]
    #             observations_features = self.get_discrete_state(ram_next)
    #             next_action = self.policy(observations_features)

    #             q_current = np.dot(self.weights, self.get_state_action_feature(state_features, action))
    #             q_next = np.dot(self.weights, self.get_state_action_feature(observations_features, next_action))
    #             td_error = reward + self.gamma * q_next - q_current

    #             if np.isfinite(td_error):
    #                 sa_features = self.get_state_action_feature(state_features, action)
    #                 self.weights += self.alpha * td_error * sa_features
                
    #             state_features, action = observations_features, next_action
    #             total_reward += reward
    #             episode_step += 1

    #             if terminal:
    #                 break
            
    #         episode_rewards.append(total_reward)
    #         self.epsilon = max(0.01, self.epsilon * self.epsilon_decay) # Epsilon decay

    #         if episode % 100 == 0:  # Checkpoint every 100 episodes
    #             self.save_checkpoint(episode)

    #         print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {episode_step}")
        
    #     plot_rewards(episode_rewards)
    #     return episode_rewards
    
    def get_state_action_feature(self, state_features, action):
        """
        Constructs a feature vector for a given action by placing the state_features
        into the section of the vector corresponding to that action.
        """
        feature_vector = np.zeros(self.num_actions * len(state_features))
        action_offset = action * len(state_features)
        feature_vector[action_offset:action_offset + len(state_features)] = state_features
        return feature_vector


    def policy(self, state_features):
        """Select action using epsilon-greedy for double Q-learning"""
        # Epsilon greedy policy
        # Return random action if rand num between 0 and 1 is less than epsilon
        if np.random.rand() < self.epsilon:
            random_action = np.random.choice(self.actions)
            return random_action
        # Else find action with the highest q value
        else:
            max_q_value = -float('inf')
            best_action = np.random.choice(self.actions)
            for action in self.actions:
                # Compute Q-values from both policies (theta1 and theta2)
                q_value_theta1 = np.dot(self.theta1, self.get_state_action_feature(state_features, action))
                q_value_theta2 = np.dot(self.theta2, self.get_state_action_feature(state_features, action))

                # Sum the Q-values from both policies for each action
                total_q_value = (q_value_theta1 + q_value_theta2) / 2
                if total_q_value > max_q_value:
                    best_action = action
                    max_q_value = total_q_value
            return best_action
    
    def extract_state_feature(self, ram_data):
        """Gets the player positions from the ram data"""
        player_x = int(ram_data[32])
        player_y = int(ram_data[34])
        opponent_x = int(ram_data[33])
        opponent_y = int(ram_data[35])
        dx = opponent_x - player_x 
        dy = opponent_y - player_y
        state = [player_x, player_y, opponent_x, opponent_y]
        return state
    

    def double_learn(self, episodes=500):
        """Performs double q learning"""
        episode_rewards = []  # store rewards

        for episode in range(episodes):
            observations, _ = self.env.reset()
            ram = observations[self.ID]
            # Use extract_state_feature to get player positions
            state_features = self.extract_state_feature(ram)
            action = self.policy(state_features)  # Select first action using epsilon greedy
            total_reward = 0
            episode_step = 0

            terminal = False
            while not terminal:
                # Multi-agent envs need an action dictionary to work
                actions = {}
                for agent in self.env.agents:
                    if agent == self.ID:
                        actions[agent] = action
                    else:
                        actions[agent] = np.random.choice(self.actions)
                
                observations, rewards, terminations, truncations, _ = self.env.step(actions)
                terminal = terminations[self.ID] or truncations[self.ID]
        
                # Extract reward and next state from ram data
                reward = rewards[self.ID]
                ram_next_state = observations[self.ID]

                # Use extract_state_feature for position-based features
                next_state_features = self.extract_state_feature(ram_next_state)
                next_action = self.policy(next_state_features)

                # Create experience and add it to experience array
                experience = (state_features, action, reward, next_state_features, terminal)
                self.experiences.append(experience)

                # Experience replay
                if len(self.experiences) >= self.sample_size:
                    random_sample = random.sample(self.experiences, self.sample_size)
                    
                    for exp in random_sample:
                        sample_state, sample_action, sample_reward, sample_next_state, sample_terminal = exp
                        
                        if np.random.rand() < 0.5:

                            # Use theta1 to identify the action with the highest q value
                            q_vals = []
                            for action in self.actions:
                                q_val = np.dot(self.theta1, self.get_state_action_feature(sample_next_state, action))
                                q_vals.append(q_val)
                            best_action = self.actions[np.argmax(q_vals)]

                            # Use other set of weights (theta 2) to calculate target value - used to update theta 1
                            if sample_terminal: 
                                target = sample_reward 
                            else:
                                target = sample_reward + self.gamma * np.dot(self.theta2, self.get_state_action_feature(sample_next_state, best_action))
                            
                            # Calculate the current q value for theta 1
                            q_current = np.dot(self.theta1, self.get_state_action_feature(sample_state, sample_action))
                            # Error here is difference between target from theta 2 and current from theta 1
                            td_error = target - q_current
                            # Update theta 1 using this error
                            self.theta1 += self.alpha * td_error * self.get_state_action_feature(sample_state, sample_action)
                        else:
                            # Use theta2 to identify the action with the highest q value
                            q_vals = []
                            for action in self.actions:
                                q_val = np.dot(self.theta2, self.get_state_action_feature(sample_next_state, action))
                                q_vals.append(q_val)
                            best_action = self.actions[np.argmax(q_vals)]

                            # Use other set of weights (theta 1) to calculate target value - used to update theta 2
                            if sample_terminal: 
                                target = sample_reward + 0
                            else:
                                target = sample_reward + self.gamma * np.dot(self.theta1, self.get_state_action_feature(sample_next_state, best_action))

                            # Calculate the current q value for theta 2
                            q_current = np.dot(self.theta2,  self.get_state_action_feature(sample_state, sample_action))
                            # Error here is difference between target from theta 1 and current from theta 2
                            td_error = target - q_current
                            # Update theta 2 using this error
                            self.theta2 += self.alpha * td_error * self.get_state_action_feature(sample_state, sample_action)


                # Shift current state and action along + add reward to total reward
                state_features = next_state_features
                action = next_action
                total_reward += reward
                episode_step += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        self.plot_q_learning_learning_curve(episode_rewards)
        return episode_rewards

    def plot_q_learning_learning_curve(self, rewards):
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("CartPole Double Q-Learning + Tile Coding")
        plt.grid(True)
        plt.show()


    def save_checkpoint(self, episode):
        os.makedirs('checkpoints', exist_ok=True)
        np.save(f'checkpoints/weights_episode_{episode}.npy', self.weights)

    def plot_rewards(self, rewards_per_episode):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Curve - Total Reward per Episode')
        plt.grid(True)
        plt.legend()
        plt.savefig('learning_curve.png')
        plt.show()

# TileCoder class remains unchanged

# Set up tile coder and agent

agent = Standard_Agent()
agent.double_learn(5000)
