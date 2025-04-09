import numpy as np
import cv2
import matplotlib.pyplot as plt
from pettingzoo.atari import boxing_v2
import random
import numpy as np
from collections import defaultdict
from PIL import Image
import seaborn as sns

class TileCoder:
    """
    Class for creating tile_coding instances.
    
    Inputs:

    min_value -> An array containing the minimum possible value for each feature in the state i.e. [min_player_x, min_player_y, min_opponent_x....]
    max_value -> An array containing the maximum possible value for each feature in the state i.e. [max_player_x, max_player_y, max_opponent_x....]
    bins -> An array specifying how many 'buckets' there should be for each state. Essentially is how many splits you have for each state feature
            e.g. if bins = [10, 10, 10, 10, 10, 10] each state feature will have 10 possible discrete values 
    num_tile_grids -> The total number of tile grids, each of which will be offset slightly

    Creating an instance of the class uses the above inputs to create N tile grids (N = num_tile_grids). Each of which are offset slightly.
    Using this collection of tile grids, the continuous state is made discrete using the discretise method. This discrete state is then extracted using the get_feature_vector method to be used in the algorithm of choice.

    An example usage using RAM render approach would be:

    tile_coder = TileCoder(...)
    discrete_state = tile_coder.get_feature_vector(ram_state)
    < Use discrete state in algo >

    This would return a discrete state which could be used for training an agent
    """
    def __init__(self, min_value, max_value, bins, num_tile_grids):
        # Ensure these are arrays
        self.min_value = np.array(min_value) 
        self.max_value = np.array(max_value) 
        self.bins = np.array(bins) 
        self.num_tile_grids = num_tile_grids 
        
        # Each tile within a grid will have the same width
        self.tile_width = (self.max_value - self.min_value) / self.bins 

        self.tiles = []
        # Create the necessary number of tiles based on self.num_tile_grids then add them to the self.tiles list
        for i in range(self.num_tile_grids):
            # For each tile grid, apply a different offset (1.5 to add a larger offset)
            offset = (i * 1.5 /self.num_tile_grids) * self.tile_width
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
            tile_boundaries = np.linspace(low, high, self.bins[i] + 1)
            tile_grid.append(tile_boundaries)
        return tile_grid
    
    def discretise(self, state, grid):
        """
        Function that finds the discrete location of each state feature in one tile grid.
        """
        discrete_location = []
        
        # For each feature in the state (player position etc.)
        for i, value in enumerate(state):
            # Find the array containing the tile cut-off values for that feature
            tile_cut_off_values = grid[i]
            
            # First check if value is outside tile range
            if value <= tile_cut_off_values[0]:
                # If value is less than first boundary, add a 0
                discrete_location.append(0)
            elif value >= tile_cut_off_values[-1]:
                # If the value is too large, add it to the last valid tile index
                discrete_location.append(len(tile_cut_off_values) - 2)
            else:
                # Find which tile/bin this value falls into
                for j in range(len(tile_cut_off_values) - 1):
                    # If the value lies within the current tile, then add it to the discrete location
                    if tile_cut_off_values[j] <= value < tile_cut_off_values[j + 1]:
                        discrete_location.append(j)
                        break
        
        return tuple(discrete_location)

    
    def get_feature_vector(self, state):
        """
        Function that takes in the state and returns a binary feature vector that represents the discretised state
        For each tile grid, the function calculates which tile the state is in and sets this index in the feature vector to 1
        """
        state = np.array(state)
        
        # Calculate total size of one tile grid and then all tile grids (bins per dimension * number of tile grids)
        total_bins = np.prod(self.bins) # Total number of possible tiles per grid
        feature_vec = np.zeros(self.num_tile_grids * total_bins) # Total number of tile grids * total number of tiles

        # For each tile grid, find where the state lies and then combine these into one vector
        for tiling_idx, grid in enumerate(self.tiles):
            # Get the index for the given state in this tile grid
            bin_indices = self.discretise(state, grid)
            
            # Need to go from a multi-dimensional bin index e.g. (1,2,3) to one dimensional e.g. (6)
            flat_idx = 0
            multiplier = 1 # Needed to shift index for each different tile grid
            # Loop through each bin index starting from the last dimension
            for dim_idx in reversed(range(len(bin_indices))):
                bin_index = bin_indices[dim_idx]
                flat_idx += bin_index * multiplier
                if dim_idx > 0: # If it isn't the first dimension, update the multiplier
                    multiplier *= self.bins[dim_idx-1]
            
            # Calculate the final index in the feature vector for this tile grid
            feature_idx = tiling_idx * total_bins + flat_idx
            feature_vec[feature_idx] = 1.0 # Set the feature to 1

        return feature_vec

#############################################################################
############# CODE TO CHECK THIS WORKS - CAN DELETE/COMMENT OUT #############
#############################################################################

# env = boxing_v2.parallel_env(render_mode="human", obs_type="ram")
# observations = env.reset()

# # Player position RAM addresses for Boxing
# # RAM indices based on Atari documentation
# PLAYER1_X_RAM_IDX = 32  # Player's x-position
# PLAYER1_Y_RAM_IDX = 34  # Player's y-position
# PLAYER2_X_RAM_IDX = 33  # Opponent's x-position
# PLAYER2_Y_RAM_IDX = 35  # Opponent's y-position


# # Define the boundaries for the tile coder
# # Boxing's playfield is approximately in these ranges:
# min_value = [0, 0, 0, 0, -120, -120]  # min_player_x, min_player_y, min_opponent_x, min_opponent_y
# max_value = [120, 120, 120, 120, 120, 120]  # max values for each position

# # Create a tile coder with 10x10 bins in each dimension and 4 tilings
# bins = [20, 20, 20, 20, 20, 20]  # divide each dimension into 10 bins
# num_tilings = 10
# tile_coder = TileCoder(min_value, max_value, bins, num_tilings)

# # Initialize dictionary to track visited states
# visited_states = defaultdict(int)
# player_positions = []
# player_tile_positions = []

# # Number of steps to run
# num_steps = 2000

# # Run the environment and collect data
# for step in range(num_steps):
#     # Take random actions
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
#     # Step the environment
#     observations, rewards, terminations, truncations, infos = env.step(actions)
    
#     # Extract RAM data from first_0
#     ram_data = observations['first_0']
    
#     # Extract and convert RAM values to signed integers FIRST
#     player_x = np.int8(ram_data[PLAYER1_X_RAM_IDX]).astype(np.int16)
#     player_y = np.int8(ram_data[PLAYER1_Y_RAM_IDX]).astype(np.int16)
#     opponent_x = np.int8(ram_data[PLAYER2_X_RAM_IDX]).astype(np.int16)
#     opponent_y = np.int8(ram_data[PLAYER2_Y_RAM_IDX]).astype(np.int16)

#     # Now calculate deltas (will be int16, preventing overflow)
#     dx = opponent_x - player_x
#     dy = opponent_y - player_y

#     # Clip to expected boxing ring bounds
#     dx = np.clip(dx, -120, 120)
#     dy = np.clip(dy, -120, 120)
    
    
#     # Store raw player positions for visualization
#     player_positions.append([player_x, player_y, opponent_x, opponent_y, dx, dy])
    
#     # Get the tile-coded state
#     state = [player_x, player_y, opponent_x, opponent_y, dx, dy]
    
#     # For visualization purposes, we'll just use the first tiling's discretized state
#     tile_position = tile_coder.discretise(state, tile_coder.tiles[0])
    
#     # Flatten the tile position before appending to the list
#     player_tile_positions.append(np.array(tile_position))
    
#     # Track visited states in the tile space
#     for tiling_idx, grid in enumerate(tile_coder.tiles):
#         bin_indices = tile_coder.discretise(state, grid)
#         visited_states[bin_indices] += 1
    
#     # Reset if the game is done
#     if any(terminations.values()) or any(truncations.values()):
#         observations = env.reset()

# # Convert player positions and tile positions to numpy arrays
# player_positions = np.array(player_positions)
# player_tile_positions = np.array(player_tile_positions)

# print("Player Y min/max:", np.min(player_positions[:,1]), np.max(player_positions[:,1]))
# print("Opponent Y min/max:", np.min(player_positions[:,3]), np.max(player_positions[:,3]))
# print("Delta Y min/max:", np.min(player_positions[:,5]), np.max(player_positions[:,5]))
# print("Delta X min/max:", np.min(player_positions[:,4]), np.max(player_positions[:,4]))
# # Create heatmap of visited states
# def plot_visited_states():
#     # Create a 2D grid to visualize player and opponent positions
#     # We'll use the first two dimensions (player x, y) for this example
#     heatmap = np.zeros((bins[0], bins[1]))
    
#     # Fill the heatmap with counts
#     for pos in player_tile_positions:
#         if 0 <= pos[0] < bins[0] and 0 <= pos[1] < bins[1]:
#             heatmap[pos[0], pos[1]] += 1
    
#     # Plot the heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(heatmap, annot=False, cmap='viridis', cbar=True)
#     plt.title('Player Position Heatmap in Tile Space')
#     plt.xlabel('Player X Tile')
#     plt.ylabel('Player Y Tile')
#     plt.show()
    
#     # Create a scatter plot of raw positions
#     plt.figure(figsize=(10, 8))
#     plt.scatter(player_positions[:, 0], player_positions[:, 1], 
#                 alpha=0.5, s=5, c='blue', label='Player')
#     plt.scatter(player_positions[:, 2], player_positions[:, 3], 
#                 alpha=0.5, s=5, c='red', label='Opponent')
#     plt.title('Raw Player Positions')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Visualize tile boundaries for the first tiling
#     plt.figure(figsize=(10, 8))
    
#     # Plot tile grid for player position (first 2 dimensions)
#     grid = tile_coder.tiles[0]
    
#     # Plot vertical lines for tile boundaries in x dimension
#     for x in grid[0]:
#         plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    
#     # Plot horizontal lines for tile boundaries in y dimension
#     for y in grid[1]:
#         plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)
    
#     # Plot player positions
#     plt.scatter(player_positions[:, 0], player_positions[:, 1], 
#                 alpha=0.5, s=5, c='blue', label='Player')
    
#     plt.title('Player Positions with Tile Boundaries (First Tiling)')
#     plt.xlabel('X Position')
#     plt.ylabel('Y Position')
#     plt.legend()
#     plt.grid(False)
#     plt.show()


# plot_visited_states()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import Rectangle
# import seaborn as sns
# from matplotlib import cm

# # Enhanced visualization functions
# def plot_state_to_tile_mapping(tile_coder, player_positions):
#     """Visualize how continuous states map to discrete tiles"""
#     plt.figure(figsize=(15, 10))
    
#     # Plot the first tiling's grid
#     grid = tile_coder.tiles[0]
    
#     # Plot vertical and horizontal lines for tile boundaries
#     for x in grid[0]:
#         plt.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
#     for y in grid[1]:
#         plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
#     # Plot sample of player positions with their tile mappings
#     sample_indices = np.random.choice(len(player_positions), size=100, replace=False)
    
#     for idx in sample_indices:
#         state = player_positions[idx]
#         tile_state = tile_coder.discretise(state, tile_coder.tiles[0])
        
#         # Get tile boundaries
#         x_min = grid[0][tile_state[0]]
#         x_max = grid[0][tile_state[0]+1]
#         y_min = grid[1][tile_state[1]]
#         y_max = grid[1][tile_state[1]+1]
        
#         # Draw rectangle for the tile
#         plt.gca().add_patch(Rectangle(
#             (x_min, y_min), x_max-x_min, y_max-y_min,
#             fill=False, edgecolor='red', alpha=0.7, lw=2
#         ))
        
#         # Draw line from point to tile center
#         plt.plot([state[0], (x_min+x_max)/2], 
#                  [state[1], (y_min+y_max)/2], 
#                  'r-', alpha=0.3)
    
#     # Plot the actual positions
#     plt.scatter(player_positions[:, 0], player_positions[:, 1], 
#                 alpha=0.7, s=30, c='blue', label='Player Positions')
    
#     plt.title('Continuous State to Discrete Tile Mapping\n(First Tiling Only)', fontsize=14)
#     plt.xlabel('X Position', fontsize=12)
#     plt.ylabel('Y Position', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.grid(False)
#     plt.show()

# def plot_state_distribution_comparison(player_positions, player_tile_positions, bins):
#     """Compare raw vs tile-coded state distributions"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
#     # Raw positions
#     h = ax1.hist2d(player_positions[:, 0], player_positions[:, 1], 
#                    bins=50, cmap='viridis', range=[[0, 120], [0, 120]])
#     ax1.set_title('Raw Position Distribution', fontsize=14)
#     ax1.set_xlabel('X Position', fontsize=12)
#     ax1.set_ylabel('Y Position', fontsize=12)
#     fig.colorbar(h[3], ax=ax1, label='Count')
    
#     # Tile-coded positions
#     h = ax2.hist2d(player_tile_positions[:, 0], player_tile_positions[:, 1], 
#                    bins=bins[:2], cmap='viridis', range=[[0, bins[0]], [0, bins[1]]])
#     ax2.set_title('Tile-Coded Position Distribution', fontsize=14)
#     ax2.set_xlabel('X Tile Index', fontsize=12)
#     ax2.set_ylabel('Y Tile Index', fontsize=12)
#     fig.colorbar(h[3], ax=ax2, label='Count')
    
#     plt.tight_layout()
#     plt.show()

# def plot_tile_activation_examples(tile_coder, player_positions):
#     """Show examples of how states activate specific tiles"""
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     axes = axes.flatten()
    
#     # Show 6 example states
#     for i, ax in enumerate(axes[:6]):
#         idx = np.random.randint(len(player_positions))
#         state = player_positions[idx]
        
#         # Get which tiles are activated across all tilings
#         feature_vec = tile_coder.get_feature_vector(state)
        
#         # Plot the state position
#         for tiling_idx, grid in enumerate(tile_coder.tiles[:3]):  # Show first 3 tilings
#             tile_state = tile_coder.discretise(state, grid)
            
#             # Get tile boundaries
#             x_min = grid[0][tile_state[0]]
#             x_max = grid[0][tile_state[0]+1]
#             y_min = grid[1][tile_state[1]]
#             y_max = grid[1][tile_state[1]+1]
            
#             # Draw rectangle for the tile
#             ax.add_patch(Rectangle(
#                 (x_min, y_min), x_max-x_min, y_max-y_min,
#                 fill=False, edgecolor=cm.viridis(tiling_idx/3), 
#                 alpha=0.7, lw=2, label=f'Tiling {tiling_idx+1}'
#             ))
        
#         # Plot the actual position
#         ax.scatter(state[0], state[1], color='red', s=100, label='State')
        
#         ax.set_title(f'Example {i+1}: State {state[:2]}\nActivates {np.sum(feature_vec)} tiles', fontsize=10)
#         ax.set_xlim(0, 120)
#         ax.set_ylim(0, 120)
#         ax.legend(fontsize=8)
    
#     plt.tight_layout()
#     plt.suptitle('How Individual States Activate Tiles Across Multiple Tilings', y=1.02, fontsize=14)
#     plt.show()

# # Generate all enhanced visualizations
# plot_state_to_tile_mapping(tile_coder, player_positions)
# plot_state_distribution_comparison(player_positions, player_tile_positions, bins)
# plot_tile_activation_examples(tile_coder, player_positions)

# def plot_feature_mappings(tile_coder, player_positions, feature_pairs):
#     """
#     Visualize continuous-to-discrete mapping for each specified feature pair
#     feature_pairs: List of tuples specifying which features to plot against each other
#                    e.g. [(0,1), (2,3)] would plot feature 0 vs 1 and feature 2 vs 3
#     """
#     num_pairs = len(feature_pairs)
#     cols = 2  # Number of columns in subplot grid
#     rows = (num_pairs + cols - 1) // cols  # Calculate needed rows
    
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
#     if num_pairs > 1:
#         axes = axes.flatten()
#     else:
#         axes = [axes]  # Make it iterable even for single plot
    
#     feature_names = [
#         "Player X", "Player Y", 
#         "Opponent X", "Opponent Y",
#         "Delta X", "Delta Y"
#     ]
    
#     for i, (f1, f2) in enumerate(feature_pairs):
#         ax = axes[i]
#         grid = tile_coder.tiles[0]  # Use first tiling for visualization
        
#         # Plot tile boundaries for these features
#         for x in grid[f1]:
#             ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
#         for y in grid[f2]:
#             ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        
#         # Plot sample of states with their tile mappings
#         sample_indices = np.random.choice(len(player_positions), size=50, replace=False)
        
#         for idx in sample_indices:
#             state = player_positions[idx]
#             tile_state = tile_coder.discretise(state, tile_coder.tiles[0])
            
#             # Get tile boundaries for these features
#             x_min = grid[f1][tile_state[f1]]
#             x_max = grid[f1][tile_state[f1]+1]
#             y_min = grid[f2][tile_state[f2]]
#             y_max = grid[f2][tile_state[f2]+1]
            
#             # Draw rectangle for the tile
#             ax.add_patch(Rectangle(
#                 (x_min, y_min), x_max-x_min, y_max-y_min,
#                 fill=False, edgecolor='red', alpha=0.7, lw=1
#             ))
            
#             # Draw line from point to tile center
#             ax.plot([state[f1], (x_min+x_max)/2], 
#                      [state[f2], (y_min+y_max)/2], 
#                      'r-', alpha=0.2)
        
#         # Plot the actual positions
#         ax.scatter(player_positions[:, f1], player_positions[:, f2], 
#                     alpha=0.7, s=15, c='blue')
        
#         ax.set_title(f'{feature_names[f1]} vs {feature_names[f2]}', fontsize=12)
#         ax.set_xlabel(f'{feature_names[f1]}', fontsize=10)
#         ax.set_ylabel(f'{feature_names[f2]}', fontsize=10)
#         ax.grid(False)
    
#     # Hide any unused subplots
#     for j in range(i+1, len(axes)):
#         axes[j].axis('off')
    
#     plt.tight_layout()
#     plt.suptitle('Continuous-to-Discrete Mapping for Feature Pairs\n(First Tiling Only)', y=1.02, fontsize=14)
#     plt.show()

# # Define which feature pairs to visualize
# feature_pairs = [
#     (0, 1),  # Player X vs Player Y
#     (2, 3),  # Opponent X vs Opponent Y
#     (0, 2),  # Player X vs Opponent X
#     (1, 3),  # Player Y vs Opponent Y
#     (4, 5)   # Delta X vs Delta Y
# ]

# plot_feature_mappings(tile_coder, player_positions, feature_pairs)


# def plot_state_discretization_summary(tile_coder, player_positions):
#     """
#     Creates a comprehensive visualization showing:
#     1. Raw state distribution
#     2. Tile boundaries and assignments
#     3. Points that fall outside tile boundaries
#     4. Summary statistics
#     """
#     plt.figure(figsize=(15, 10))
    
#     # Use first tiling for visualization
#     grid = tile_coder.tiles[0]
    
#     # Plot tile boundaries
#     for x in grid[0]:
#         plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5, label='Tile boundaries' if x == grid[0][0] else "")
#     for y in grid[1]:
#         plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)
    
#     # Track points that fall outside tiles
#     outside_points = []
#     assigned_points = []
    
#     # Sample points for visualization (don't plot all to avoid clutter)
#     sample_indices = np.random.choice(len(player_positions), size=100, replace=False)
    
#     for idx in sample_indices:
#         state = player_positions[idx]
#         tile_state = tile_coder.discretise(state, tile_coder.tiles[0])
        
#         # Check if point falls within tile grid
#         x_in_bounds = (grid[0][0] <= state[0] <= grid[0][-1])
#         y_in_bounds = (grid[1][0] <= state[1] <= grid[1][-1])
        
#         if not (x_in_bounds and y_in_bounds):
#             outside_points.append(state)
#             continue
            
#         assigned_points.append(state)
        
#         # Get tile boundaries
#         x_min = grid[0][tile_state[0]]
#         x_max = grid[0][tile_state[0]+1]
#         y_min = grid[1][tile_state[1]]
#         y_max = grid[1][tile_state[1]+1]
        
#         # Draw tile rectangle
#         plt.gca().add_patch(Rectangle(
#             (x_min, y_min), x_max-x_min, y_max-y_min,
#             fill=False, edgecolor='red', alpha=0.7, lw=1,
#             label='Assigned tile' if idx == sample_indices[0] else ""
#         ))
        
#         # Draw line from point to tile center
#         plt.plot([state[0], (x_min+x_max)/2], 
#                  [state[1], (y_min+y_max)/2], 
#                  'r-', alpha=0.3)
    
#     # Convert to arrays for plotting
#     if assigned_points:
#         assigned_points = np.array(assigned_points)
#         plt.scatter(assigned_points[:, 0], assigned_points[:, 1], 
#                     alpha=0.7, s=30, c='blue', label='Assigned points')
    
#     if outside_points:
#         outside_points = np.array(outside_points)
#         plt.scatter(outside_points[:, 0], outside_points[:, 1], 
#                     alpha=0.7, s=50, c='black', marker='x', label='Outside tile grid')
    
#     # Add summary text
#     summary_text = (
#         f"Total points: {len(player_positions)}\n"
#         f"Points outside tile grid: {len(outside_points)}\n"
#         f"Tile grid coverage: {100*(1-len(outside_points)/len(player_positions)):.1f}%\n"
#         f"Tile grid x-range: [{grid[0][0]:.1f}, {grid[0][-1]:.1f}]\n"
#         f"Tile grid y-range: [{grid[1][0]:.1f}, {grid[1][-1]:.1f}]"
#     )
#     plt.gcf().text(0.15, 0.85, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.title('State-to-Discrete Transition Summary\n(First Tiling Only)', fontsize=14)
#     plt.xlabel('X Position', fontsize=12)
#     plt.ylabel('Y Position', fontsize=12)
#     plt.legend(fontsize=10, loc='upper right')
#     plt.grid(False)
#     plt.show()

# # Generate the summary visualization
# plot_state_discretization_summary(tile_coder, player_positions)

# def plot_all_tilings(tile_coder, player_positions):
#     """Plot all tilings in a grid layout showing their different partitions"""
#     num_tilings = len(tile_coder.tiles)
#     cols = int(np.ceil(np.sqrt(num_tilings)))
#     rows = int(np.ceil(num_tilings / cols))
    
#     fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
#     if num_tilings > 1:
#         axes = axes.flatten()
#     else:
#         axes = [axes]
    
#     # Plot each tiling in its own subplot
#     for tiling_idx in range(num_tilings):
#         ax = axes[tiling_idx]
#         grid = tile_coder.tiles[tiling_idx]
        
#         # Plot tile boundaries
#         for x in grid[0]:
#             ax.axvline(x=x, color='gray', linestyle='--', alpha=0.7)
#         for y in grid[1]:
#             ax.axhline(y=y, color='gray', linestyle='--', alpha=0.7)
        
#         # Plot player positions
#         ax.scatter(player_positions[:, 0], player_positions[:, 1],
#                   alpha=0.5, s=10, c='blue')
        
#         # Add tiling info
#         offset = (tiling_idx / num_tilings) * tile_coder.tile_width
#         ax.set_title(f'Tiling {tiling_idx+1}\nOffset: [{offset[0]:.1f}, {offset[1]:.1f}]')
#         ax.set_xlabel('X Position')
#         ax.set_ylabel('Y Position')
#         ax.set_xlim(tile_coder.min_value[0], tile_coder.max_value[0])
#         ax.set_ylim(tile_coder.min_value[1], tile_coder.max_value[1])
    
#     # Hide any unused subplots
#     for i in range(num_tilings, rows*cols):
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     plt.suptitle(f'All {num_tilings} Tilings with Their Unique Partitions', y=1.02, fontsize=14)
#     plt.show()

# # Generate the tiling visualization
# plot_all_tilings(tile_coder, player_positions)