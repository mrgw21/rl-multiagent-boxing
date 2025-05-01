import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import ale_py
from abc import ABC, abstractmethod
from utils.tile_coding import TileCoder
from utils.agent_utils import save_linear_agent, load_linear_agent, test_linear_agent, plot_rewards
from utils.PER import store_experience, prioritised_sample, update_priority_order

class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.reset()
        self.actions = [0, 1]  # Two actions: left and right 
        self.num_actions = len(self.actions)

        # Hyperparameters
        self.alpha = 0.01
        self.alpha_decay = 0.995
        self.min_alpha = 0.001
        self.epsilon = 0.15
        self.gamma = 0.9
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01

        # Parameters for experience replay
        self.replay_buffer = deque(maxlen=1000)
        self.priorities = deque(maxlen=1000)
        self.exp_alpha = 0.6  # Priority exponent
        self.exp_beta = 0.4   # Importance sampling exponent
        self.batch_size = 32

        #Place holders for weights of both networks
        self.theta1 = None
        self.theta2 = None

        # Add additional target networks to help stabilise training
        self.theta1_target = np.copy(self.theta1)
        self.theta2_target = np.copy(self.theta2)
        self.step_update = 100 # update every 100 steps

    def update_target_networks(self):
        """Update target weights with current weights"""
        self.theta1_target = np.copy(self.theta1)
        self.theta2_target = np.copy(self.theta2)
    
    def policy(self, state_features, greedy = False):
        """Select action using epsilon-greedy for double Q-learning"""
        # Epsilon greedy policy
        # Return random action if rand num between 0 and 1 is less than epsilon
        if np.random.rand() < self.epsilon and not greedy:
            random_action = np.random.choice(self.actions)
            return random_action
        # Else find action with the highest q value
        else:
            q_vals = []
            for action in self.actions:
                # Compute Q-values from both policies (theta1 and theta2)
                q_value_theta1 = np.dot(self.theta1, self.get_state_action_feature(state_features, action))
                q_value_theta2 = np.dot(self.theta2, self.get_state_action_feature(state_features, action))

                # Sum the Q-values from both policies for each action
                total_q_value = (q_value_theta1 + q_value_theta2) / 2
                q_vals.append(total_q_value)
            best_action = self.actions[np.argmax(q_vals)]
            return best_action
        
    def get_state_action_feature(self, state_features, action):
        """Creates a state feature vector for each action. Essentially joins state feature vectors for each action together.
        i.e. if we have 8 actions, we will have 8 state feature vectors, in order from 1 to 8"""
        feature_vector = np.zeros(self.num_actions * len(state_features)) # Number of actions * number of state features
        action_offset = action * len(state_features) # Vector for each action will be offset by the length of original state feature
        # For the provided action, set that part of the feature vector to the state features provided
        feature_vector[action_offset:action_offset + len(state_features)] = state_features 
        return feature_vector
    
    def double_learn_no_PER(self, num_episodes):
        """Run double q learning with experience replay but not prioritised experience replay"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Gym envs return observations and info - we only need observations
            observations, _ = self.env.reset()
            # Extract the state feature from above using tile code
            state_features = self.feature_extraction(observations)
            action = self.policy(state_features) # First action based on policy

            # Set initial w vars to track rewards + game over
            total_reward = 0
            episode_steps = 0
            terminal = False
            truncated = False

            # In this env, the game is over if terminal or truncated become true
            while not (terminal or truncated):
                
                if episode_steps % self.step_update == 0:
                    self.update_target_networks()

                next_state, reward, terminal, truncated, _ = self.env.step(action) # Same again for info (_)
                next_state_features = self.feature_extraction(next_state)
                next_action = self.policy(next_state_features)

                # Store every experience for replay
                experience = (state_features, action, reward, next_state_features, terminal or truncated)
                self.replay_buffer.append(experience)

                # If there are enough experiences, use a batch from the array to update the Q tables
                if len(self.replay_buffer) >= self.batch_size:
                    random_sample = random.sample(self.replay_buffer, self.batch_size) # Randomly select batch_size experiences
                    # For each experience in the randomly selected samples
                
                    for sample_state, sample_action, sample_reward, sample_next_state, sample_terminal in random_sample:
                        # Double Q-learning - randomly chooses between updating one function or the other
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
                                target = sample_reward + self.gamma * np.dot(self.theta2_target, self.get_state_action_feature(sample_next_state, best_action))
                            
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
                                target = sample_reward + self.gamma * np.dot(self.theta1_target, self.get_state_action_feature(sample_next_state, best_action))

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
                episode_steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        plot_rewards(episode_rewards)
        return episode_rewards
    
    def double_learn_PER(self, num_episodes):
        """Run double q learning with prioritised experience replay"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Gym envs return observations and info - we only need observations
            observations, _ = self.env.reset()
            # Extract the state feature from above using tile code
            state_features = self.feature_extraction(observations)
            action = self.policy(state_features) # First action based on policy

            # Set initial w vars to track rewards + game over
            total_reward = 0
            episode_steps = 0
            terminal = False
            truncated = False

            # In this env, the game is over if terminal or truncated become true
            while not (terminal or truncated):
                
                if episode_steps % self.step_update == 0:
                    self.update_target_networks()

                next_state, reward, terminal, truncated, _ = self.env.step(action) # Same again for info (_)
                next_state_features = self.feature_extraction(next_state)
                next_action = self.policy(next_state_features)

                # Calculate the TD error based on current weights 
                q_vals = [np.dot(self.theta1, self.get_state_action_feature(state_features, action)) for action in self.actions]
                current_q = np.dot(self.theta1, self.get_state_action_feature(state_features, action))
                if (terminal or truncated):
                    target = reward
                else:
                    target = reward + self.gamma * max(q_vals)
                td_error = target - current_q
            
                # Store every experience for replay
                experience = (state_features, action, reward, next_state_features, terminal or truncated)
                store_experience(self, experience, abs(td_error), cache=False)


                # If there are enough experiences, use a batch from the array to update the Q tables
                if len(self.replay_buffer) >= self.batch_size:
                    sample_batch, indices, weights = prioritised_sample(self)
                    # For each experience in the randomly selected samples
                
                    td_errors = []
                    
                    for sample_experience in sample_batch:
                        sample_state, sample_action, sample_reward, sample_next_state, sample_terminal = sample_experience
                        # Double Q-learning - randomly chooses between updating one function or the other
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
                            td_errors.append(td_error)
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
                            td_errors.append(td_error)

                    update_priority_order(self, indices, td_errors)

                # Shift current state and action along + add reward to total reward
                state_features = next_state_features
                action = next_action
                total_reward += reward
                episode_steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        plot_rewards(episode_rewards)
        return episode_rewards
    
        
    @abstractmethod  
    def feature_extraction(self, state):
        pass


class TileCodedAgent(Agent):
    def __init__(self):
        super().__init__()

        # Max and mins for cart pole states - [cart position, cart velocity, pole angle, pole velocity]
        self.min_values = [-4.8, -5.0, -0.418, -5.0]
        self.max_values = [4.8, 5.0, 0.418, 5.0]
        self.bins = [10, 10, 10, 10]  # 10 bins for each dimension
        self.num_tile_grids = 8  # Number of overlapping tile grids
        
        # Create tile coder
        self.tile_coder = TileCoder(self.min_values, self.max_values, self.bins, self.num_tile_grids)
        # Feature size is product of bins multiplied by num of tile grids
        self.feature_size = np.prod(self.bins) * self.num_tile_grids
        
        # Thetas are weights for double learning - will be a separate vector for each action
        self.theta1 = np.zeros(self.feature_size * self.num_actions)
        self.theta2 = np.zeros(self.feature_size * self.num_actions)
    
    
    def feature_extraction(self, state):
        """Get state features using the tile coder - i.e. get a discretised state from a continuous one"""
        return self.tile_coder.get_feature_vector(state)
    

# Standard agent implementation for comparison
class StandardAgent(Agent):
    def __init__(self):
        super().__init__()
        # CartPole has only four state features
        # [Cart pos., Cart vel., Pole ang., Pole ang. vel.]
        self.feature_size = 4
            
        # Thetas are weights for double learning - will be a separate vector for each action
        self.theta1 = np.zeros(self.feature_size * self.num_actions)
        self.theta2 = np.zeros(self.feature_size * self.num_actions)


    def feature_extraction(self, state):
        """No tile coding so just return the state as a numpy array"""
        state = np.array(state)
        return state


if __name__ == "__main__":
    # Create + Train agents
    standard_agent_no_PER = StandardAgent()
    standard_agent_PER = StandardAgent()
    tile_agent_no_PER = TileCodedAgent()
    tile_agent_PER = TileCodedAgent()

    standard_agent_no_PER.double_learn_no_PER(200)
    standard_agent_PER.double_learn_PER(200)
    tile_agent_no_PER.double_learn_PER(200)
    tile_agent_PER.double_learn_PER(200)

    # Save agents
    save_linear_agent(standard_agent_no_PER, "standard_agent_no_PER")
    save_linear_agent(standard_agent_PER, "standard_agent_PER")
    save_linear_agent(tile_agent_no_PER, "tile_agent_no_PER")
    save_linear_agent(tile_agent_PER, "tile_agent_PER")

    # Test agents
    test_linear_agent(standard_agent_no_PER, cartpole=True)
    test_linear_agent(standard_agent_PER, cartpole=True)
    test_linear_agent(tile_agent_no_PER, cartpole=True)
    test_linear_agent(tile_agent_PER, cartpole=True)
