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
        half_grids = self.num_tile_grids // 2 # allows new tiles to be centred around initial tiles rather than only offset in one direction
        # If num tile grids is odd
        if self.num_tile_grids % 2 == 1:
            offsets = np.arange(-half_grids, half_grids + 1)
        # Even
        else:
            offsets = np.arange(-half_grids, half_grids)

        for offset_factor in offsets:
            # For each tile grid, apply a different offset using the factors from above
            offset = (offset_factor / self.num_tile_grids) * self.tile_width
            self.tiles.append(self.create_tile(offset)) # Create the tile grid using the unique offset
    
    def create_tile(self, offset_factor):
        """
        Function that creates a new tile grid using the provided offset.
        It first updates the original min and max values with the offset and then defines the new individual tile boundaries.
        """
        offset = np.ones_like(self.min_value) * offset_factor
        tile_grid = [] # Used to store the tile boundaries - cut offs
        # For each tile in a grid apply offset and calculate new boundaries
        for i in range(len(self.bins)):
            # Shift min and max values by offset
            low = self.min_value[i] + offset[i] * self.tile_width[i]
            high = self.max_value[i] + offset[i] * self.tile_width[i]
            # Identify the new points where the tile boundaries will be (i.e. tile boundaries with offset)
            tile_boundaries = np.linspace(low, high, self.bins[i] + 1)
            tile_grid.append(tile_boundaries)
        return tile_grid
    
    def discretise(self, state, grid):
        """
        Function that finds the discrete location of each state feature in one tile grid.
        """
        if len(state) != len(grid):
            raise ValueError("State length and grid dimensions mismatch")

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
        state = np.array(state, dtype=np.float32).flatten()
        
        # Calculate total size of one tile grid and then all tile grids (bins per dimension * number of tile grids)
        total_bins = np.prod(self.bins) # Total number of possible tiles per grid
        feature_vec = np.zeros(self.num_tile_grids * total_bins) # Total number of tile grids * total number of tiles

        # For each tile grid, find where the state lies and then combine these into one vector
        for tiling_idx, grid in enumerate(self.tiles):
            # Get the index for the given state in this tile grid
            bin_indices = self.discretise(state, grid)
            
            # Convert multi-dimensional bin index to one-dimensional
            flat_idx = np.ravel_multi_index(bin_indices, self.bins)
            
            # Calculate the final index in the feature vector for this tile grid
            feature_idx = tiling_idx * total_bins + flat_idx
            feature_vec[feature_idx] = 1.0 # Set the feature to 1

        return feature_vec