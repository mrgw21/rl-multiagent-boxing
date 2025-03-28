import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # Return a random action from the action space
        return self.action_space.sample()
