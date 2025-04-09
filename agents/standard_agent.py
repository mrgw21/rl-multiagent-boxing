import numpy as np
import os
import cv2
from pettingzoo.atari import boxing_v2

class Standard_Agent:
    
    #Method for defining global variables
    def __init__(self):
        #Hyperparams
        self.feature_length = 522752 + 18               #Length of the feature + state combination
        self.weights = np.zeros(self.feature_length)
        self.alpha = 0.8
        self.epsilon = 0.15
        self.gamma = 0.9

        #Environment specific
        self.env = boxing_v2.parallel_env(obs_type = 'ram',render_mode=None)
        self.ID = "first_0"

    #Function designed to extract the list of features from the atari observation based on the type 'ram'
    def feature_extraction(self, observations):
        
        #Initial observations, which are stored as a list of 128 values, ranging from 0-256 are 'unpacked' to bits.
        binary_observations = np.unpackbits(observations)
        #Make use of the np.triu_indices function to return pairs, i and j, of all possible bit combinations, ensuring no repeats
        i, j = np.triu_indices(len(binary_observations), k=1)
        #Execute these combinations using the bianry operator '&'
        pairwise_ands = binary_observations[i] & binary_observations[j]
        #Concataenate the two into a single list
        features = np.concatenate([binary_observations, pairwise_ands])

        return features



    def model(self):
        pass

    def policy(self):
        pass

    def learn_Sarsa(self):
        pass


    def learn_TD(self):
        observations, info = self.env.reset()
        observations = observations[self.ID]
        features = self.feature_extraction(observations)