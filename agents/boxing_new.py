import numpy as np
from collections import deque
from pettingzoo.atari import boxing_v2
import random
from tile_coding import TileCoder
from abc import ABC, abstractmethod
from agent_utils import save_agent, load_agent, test_agent, compare_agents, plot_learning_curve

class Agent:
    def __init__(self):
        self.env = boxing_v2.parallel_env(obs_type='ram', render_mode=None)
        self.ID = "first_0"
        self.env.reset()
        self.actions = list(range(self.env.action_space(self.ID).n))
        self.num_actions = len(self.actions)

        # Hyperparameters
        self.alpha = 0.05
        self.alpha_decay = 0.995
        self.min_alpha = 0.001
        self.epsilon = 0.15
        self.gamma = 0.9
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        # Parameters for experience replay
        self.experiences = deque(maxlen=1000)
        self.sample_size = 32

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
    
    def double_learn(self, num_episodes):
        """Run double q learning"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Gym envs return observations and info - we only need observations
            observations, _ = self.env.reset()
            # Extract the state feature from above using tile code
            state_features = self.extract_state_feature(observations[self.ID])
            action = self.policy(state_features) # First action based on policy

            # Set initial w vars to track rewards + game over
            total_reward = 0
            episode_steps = 0
            finished = False

            # In this env, the game is over if terminal or truncated become true
            while not finished:
                # Check if target network update is needed
                if episode_steps % self.step_update == 0:
                    self.update_target_networks()
                
                
                # Create a dictionary of actions for all agents
                actions = {agent: (self.policy(self.extract_state_feature(observations[agent]))
                                if agent == self.ID else random.choice(self.actions))
                        for agent in self.env.agents}
                

                next_observations, rewards, terminal, truncated, _ = self.env.step(actions) # Same again for info (_)
                
                next_state_features = self.extract_state_feature(next_observations[self.ID])
                reward = rewards[self.ID]
                finished = terminal[self.ID] or truncated[self.ID]

                next_action = self.policy(next_state_features)
                # Store every experience for replay
                experience = (state_features, action, reward, next_state_features, finished)
                self.experiences.append(experience)

                # If there are enough experiences, use a batch from the array to update the Q tables
                if len(self.experiences) >= self.sample_size:
                    random_sample = random.sample(self.experiences, self.sample_size) # Randomly select sample_size experiences
                    # For each experience in the randomly selected samples
                
                    for sample_state, sample_action, sample_reward, sample_next_state, sample_finished in random_sample:
                        # Double Q-learning - randomly chooses between updating one function or the other
                        if np.random.rand() < 0.5:

                            # Use theta1 to identify the action with the highest q value
                            q_vals = []
                            for action in self.actions:
                                q_val = np.dot(self.theta1, self.get_state_action_feature(sample_next_state, action))
                                q_vals.append(q_val)
                            best_action = self.actions[np.argmax(q_vals)]

                            # Use other set of weights (theta 2) to calculate target value - used to update theta 1
                            if sample_finished: 
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
                            if sample_finished: 
                                target = sample_reward + 0
                            else:
                                target = sample_reward + self.gamma * np.dot(self.theta1_target, self.get_state_action_feature(sample_next_state, best_action))

                            # Calculate the current q value for theta 2
                            q_current = np.dot(self.theta2, self.get_state_action_feature(sample_state, sample_action))
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
        
        plot_learning_curve(episode_rewards)
        return episode_rewards
    
        
    @abstractmethod  
    def extract_state_features(self, ram_data):
        pass


class TileCodedAgent(Agent):
    def __init__(self):
        super().__init__()

        # Max and mins for cart pole states - [cart position, cart velocity, pole angle, pole velocity]
        self.min_values = [0.0, 0.0, 0.0, 0.0]
        self.max_values = [120.0, 120.0, 120.0, 120.0]
        self.bins = [8, 8, 8, 8]  # 10 bins for each dimension
        self.num_tile_grids = 3  # Number of overlapping tile grids
        
        # Create tile coder
        self.tile_coder = TileCoder(self.min_values, self.max_values, self.bins, self.num_tile_grids)
        # Feature size is product of bins multiplied by num of tile grids
        self.feature_size = np.prod(self.bins) * self.num_tile_grids
        
        # Thetas are weights for double learning - will be a separate vector for each action
        self.theta1 = np.zeros(self.feature_size * self.num_actions)
        self.theta2 = np.zeros(self.feature_size * self.num_actions)

        # Add additional target networks to help stabilise training
        self.theta1_target = np.copy(self.theta1)
        self.theta2_target = np.copy(self.theta2)
    
    def extract_state_feature(self, ram_data):
        """Gets the player positions from the ram data"""
        player_x = int(ram_data[32])
        player_y = int(ram_data[34])
        opponent_x = int(ram_data[33])
        opponent_y = int(ram_data[35])
        state = [player_x, player_y, opponent_x, opponent_y]
        dis_state = self.tile_coder.get_feature_vector(state)
        return dis_state
    

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

        # Add additional target networks to help stabilise training
        self.theta1_target = np.copy(self.theta1)
        self.theta2_target = np.copy(self.theta2)


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


standard_agent = StandardAgent()
tile_agent = TileCodedAgent()
standard_agent, tile_agent = compare_agents(standard_agent, tile_agent, 200)
save_agent(standard_agent, path = 'saved_agents/standard_agent.pkl')
save_agent(tile_agent, path = 'saved_agents/tile_agent.pkl')
loaded_s_agent = load_agent('saved_agents/standard_agent.pkl')
loaded_t_agent = load_agent('saved_agents/tile_agent.pkl')


test_agent(standard_agent)
test_agent(loaded_s_agent)
test_agent(tile_agent)
test_agent(loaded_t_agent)

