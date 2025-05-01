import numpy as np
import cv2
import matplotlib.pyplot as plt
from pettingzoo.atari import boxing_v2
import random
import numpy as np
from collections import defaultdict

class TileCoder:
    def __init__(self, min_value, max_value, bins, num_tile_grids):
        self.min_value = min_value # Array of minimum possible values (min_player_x, min_player_y, min_opponent_x....)
        self.max_value = max_value # Array of maximum possible values (max_player_x, max_player_y, max_opponent_x....)
        self.bins = bins # Number of tiles along x and y axis (bin_x, bin_y) - i.e. how many times to split each tile grid up
        self.num_tile_grids = num_tile_grids # How many grids will be overlapped over each other with the offset 
        
        # Each tile within a grid will have the same width
        self.tile_width = (self.max_value - self.min_value) / (self.bins - 1)
        self.tiles = []
        # For each tile grid, apply a different offset 
        for i in range(self.num_tile_grids):
            offset = (i / self.num_tile_grids) * self.tile_width
            self.tiles.append(self.create_tile(offset)) # Create the tile grid using the unique offset
    
    def create_tile(self, offset):
        """
        Function that creates a new tile grid using the provided offset.
        It first updates the original min and max values with the offset and then defines the new individual tile boundaries.
        """
        tile_grid = [] # Used to store the tile boundaries - cut offs
        # For each tile in a grid apply offset and calculate new boundaries
        for i in range(len(self.bins)):
            # Shift min and max values by offset
            low = self.min_value[i] + offset[i]
            high = self.max_value[i] + offset[i]
            # Identify the new points where the tile boundaries will be (i.e. tile boundaries with offset)
            tile_boundaries = np.linspace(low, high, self.bins[i] + 1)[1:-1]
            tile_grid.append(tile_boundaries)
        return tile_grid
    
    def discretise(self, state, grid):
        """
        Function that returns the location of a state on the given grid
        """
        discrete_location = []
        
        for value, boundary in zip(state, grid):
            bin_idx = int(np.digitize(value, boundary))
            print(bin_idx)
            discrete_location.append(bin_idx)
        
        print("Discretised: ", state,  tuple(discrete_location))
        return tuple(discrete_location)
    
    def tile_encode(self, state, tiles, flatten=False):
        """
        Function that takes a state and computes each discretised version for every tile grid.
        Flatten allows this to be returned as a single vector as opposed to N vectors, where N = number of tile grids.
        """
        discretised_states = []
        for grid in tiles:
            discretised_state = self.discretise(state, grid)
            discretised_states.append(discretised_state)
        
        if flatten:
            return np.concatenate(discretised_states)
        else:
            return discretised_states
    
    def get_feature_vector(self, state):
        """
        Function that takes in the state and returns a binary feature vector that represents the discretised state
        i.e. adds a 1 in each tile grid where the discretised state falls and then returns that for each grid
        """
        state = np.array(state)
        feature_vec = np.zeros(self.num_tile_grids * np.prod(self.bins))

        # for each tile grid, calculate the location of the discretised state in that tile
        for tile_grid_idx, grid in enumerate(self.tiles):
            tile_idx_array = self.discretise(state, grid) 
            # We need to convert this index array into a single flattened index 
            # i.e. go from idx for (x1, y1, x2, y2, dx, dy) -> N where N is one index
            # To do this, we aren't concered with the row or column, but how 'deep' the value is into the array. 
            # So we use ravel_multi_index to get the absolute/flattened index of the value
            absolute_idx = np.ravel_multi_index(tile_idx_array, self.bins)
            # Knowing the tile index for that specific grid, we can caluclate the global index across all tiles
            total_idx = tile_idx_array * np.prod(self.bins) + absolute_idx
            feature_vec[total_idx] = 1.0
        
        return feature_vec



# Create tile coder for 6D state space
min_vals = np.array([0, 0, 0, 0, -100, -100])
max_vals = np.array([100, 100, 100, 100, 100, 100])
bins = np.array([10, 10, 10, 10, 10, 10])
num_tilings = 8
tile_coder = TileCoder(min_vals, max_vals, bins, num_tilings)

obs = np.array([50, 50, 20, 30, 0, 0])
feature_vector = tile_coder.get_feature_vector(obs)

print("Feature vector shape:", feature_vector.shape)
print("Non-zero indices:", np.nonzero(feature_vector)[0])


def visualize_tilings(tilings, points=None):
    """Plot each tiling as a grid, and optionally overlay data points."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']

    fig, ax = plt.subplots(figsize=(10, 10))

    legend_lines = []
    legend_labels = []

    for i, grid in enumerate(tilings):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        # Plot vertical and horizontal lines for the grid
        for x in grid[0]:
            ax.axvline(x=x, color=color, linestyle=linestyle)
        for y in grid[1]:
            ax.axhline(y=y, color=color, linestyle=linestyle)
        # Add one legend handle per tiling
        line = ax.axhline(y=grid[1][0], color=color, linestyle=linestyle)
        legend_lines.append(line)
        legend_labels.append(f"Tiling #{i}")

    if points is not None:
        points = np.array(points)
        scatter = ax.scatter(points[:, 0], points[:, 1], color='black', marker='o', s=60, edgecolors='white', label='Points')
        legend_lines.append(scatter)
        legend_labels.append("Points")

    ax.grid(False)
    ax.set_xlim(tile_coder.min_value[0], tile_coder.max_value[0])
    ax.set_ylim(tile_coder.min_value[1], tile_coder.max_value[1])
    ax.set_title("Tilings with Points")
    ax.legend(legend_lines, legend_labels, facecolor='white', framealpha=0.9)
    return ax


example_points = [
    [20, 30],
    [50, 50],
    [80, 60],
]

visualize_tilings(tile_coder.tiles, points=example_points)
plt.show()



def get_agent_centre(mask):
    """
    Returns the centre position of the player using their mask
    """
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return np.array([0.0, 0.0])
    return coords.mean(axis=0) 

def extract_positions_from_image(obs, agent_id):
    """
    Function that takes in an image observation and an agent_id and returns the appropriate state vector for the player agent.
    """

    # Because there are black and white pixels in other parts of each image, we need to crop the image to just the ring 
    roi_x_start, roi_y_start = 25, 30
    roi_x_end, roi_y_end = 135, 180
    cropped_observation = obs[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_observation, cv2.COLOR_RGB2GRAY)
    
    # Use thresholding to determine player and opponent masks - where pixels are white and black, respectively.
    _, player_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, opponent_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Assuming the agent is the player, extract the player's position and opponent's position
    player_pos = get_agent_centre(player_mask)
    opponent_pos = get_agent_centre(opponent_mask)
    
    # Relative positions
    dx, dy = opponent_pos[1] - player_pos[1], opponent_pos[0] - player_pos[0]  

    # Return the state vector containing player positions, opponent positions, and the differences (dx, dy)
    return np.array([
        player_pos[1], player_pos[0],  # Player x, y position
        opponent_pos[1], opponent_pos[0],  # Opponent x, y position
        dx, dy  # dx, dy differences
    ], dtype=np.float32)

# Create environment
env = boxing_v2.env(obs_type="rgb_image")
agent_ids = env.possible_agents




# # Store the player's state vector over time
# player_states = []  # List to store player state vectors

# for episode in range(3):  # fewer episodes to test
#     observation, infos = env.reset()
#     step = 0
#     while env.agents and step < 100:
#         actions = {agent: random.randint(0, 17) for agent in env.agents}
#         next_observation, reward, terminated, truncated, info = env.step(actions)

#         player_agent = 'first_0'  # Can change this to second_0 if we want to get the opponents state vector
#         # Add the state vector to the list - [player_x, player_y, opponent_x, opponent_y, dx, dy]
#         if player_agent in env.agents:
#             features = extract_positions_from_image(observation[player_agent], player_agent)
#             player_states.append(features)  

#         observation = next_observation
#         step += 1

# # Convert player states to a numpy array
# player_states_array = np.array(player_states)

# # Debug: Print the player states array to see the collected data
# print("Player States array shape:", player_states_array.shape)
# print("First 5 player states:\n", player_states_array)


class LinearSarsaAgent:
    def __init__(self, num_actions, tile_coder, learning_rate=0.01, discount_factor=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.tile_coder = tile_coder
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        self.feature_size = tile_coder.total_tiles
        self.weights = np.zeros((self.num_actions, self.feature_size))

    def get_features(self, state):
        return self.tile_coder.get_features(state)

    def get_q_value(self, state, action):
        features = self.get_features(state)
        return np.dot(self.weights[action], features)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, done):
        features = self.get_features(state)
        q_current = np.dot(self.weights[action], features)

        if done:
            td_target = reward
        else:
            next_features = self.get_features(next_state)
            q_next = np.dot(self.weights[next_action], next_features)
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current
        self.weights[action] += self.alpha * td_error * features


# Function to train the agent
def train_sarsa_agent(num_episodes, agent):
    env = boxing_v2.env()
    env.reset()

    player_agent = "first_0"
    rewards_history = []

    for episode in range(num_episodes):
        env.reset()
        observation = env.observe(player_agent)
        state = extract_positions_from_image(observation, player_agent)
        action = agent.select_action(state)
        total_reward = 0
        done = False

        while not done:
            # Take the action for player_agent
            env.step(action)

            # Get environment feedback
            reward = env.rewards[player_agent]
            done = env.terminations[player_agent] or env.truncations[player_agent]
            total_reward += reward

            if not done:
                next_observation = env.observe(player_agent)
                next_state = extract_positions_from_image(next_observation, player_agent)
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
            else:
                agent.update(state, action, reward, None, None, done)

            # Let the second agent take a random action if it's their turn
            other_agent = env.agent_selection
            if other_agent == "second_0":
                env.step(np.random.randint(0, 18))  # random action for opponent

            # Optional rendering
            if env.agent_selection == player_agent:
                env.render()

        rewards_history.append(total_reward)

    env.close()
    return agent, rewards_history

# Run training
agent = LinearSarsaAgent(
    num_actions=18,
    feature_size=6,
    learning_rate=0.01,
    discount_factor=0.9,
    epsilon=0.15
)

agent, rewards = train_sarsa_agent(num_episodes=1000, agent=agent)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA Learning Curve')
plt.grid(True)
plt.show()

# Function to evaluate the trained agent
def evaluate_agent(env, agent, num_episodes=100):
    total_rewards = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        player_agent = 'first_0'
        episode_reward = 0
        done = False
        
        while not done and player_agent in env.agents:
            # Extract state
            state = extract_positions_from_image(observation[player_agent], player_agent)
            
            # Choose action with no exploration (epsilon=0)
            action = agent.select_action(state) if np.random.random() > 0.05 else np.random.randint(0, 18)
            
            # Create actions dictionary
            actions = {agent: (action if agent == player_agent else np.random.randint(0, 18)) 
                      for agent in env.agents}
            
            # Take step
            observation, rewards, terminated, truncated, _ = env.step(actions)
            
            # Update reward
            episode_reward += rewards[player_agent]
            
            # Check if done
            done = terminated[player_agent] or truncated[player_agent]
            
            # Render if desired
            env.render()
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Reward = {episode_reward}")
    
    print(f"Average Evaluation Reward: {np.mean(total_rewards):.2f}")
    return total_rewards

# This env lets us render so we can see what the agent is doing
eval_env = boxing_v2.env(obs_type="rgb_image", render_mode="human")

# Can use this to see agent's performance after training
evaluation_rewards = evaluate_agent(eval_env, agent)