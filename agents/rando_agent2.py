import numpy as np

class RandoAgent2:
    def __init__(self, action_space):
        self.action_space = action_space
        self.fixed_action = action_space.sample()  # simulate being "stuck"

    def act(self, obs):
        if np.random.rand() < 0.7:
            return self.fixed_action  # 70% chance: same action (stupid!)
        else:
            return self.action_space.sample()
