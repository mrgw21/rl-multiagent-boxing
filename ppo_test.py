import gymnasium as gym
import torch
import numpy as np
import csv
import os
import ale_py
from collections import deque
from torchvision import transforms
from training.gpu.ppo_gpu import PPOAgent
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Legacy Model Definition ===
class CNNFeatureExtractor(torch.nn.Module):
    def _init_(self):
        super()._init_()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((5, 5))
        )

    def forward(self, x):
        return self.conv_layers(x)

class ActorLegacy(torch.nn.Module):
    def _init_(self, n_actions):
        super()._init_()
        self.features = CNNFeatureExtractor()
        self.fc1 = torch.nn.Linear(128 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.policy_logits = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.policy_logits(x)


# Crucial that gymnasium is consistently used, rather than gym. Note that gymnasium is shortened to gym in import
env = gym.make("ALE/Boxing-v5", frameskip = 1, difficulty = 3) # Removes frameskip so it can be altered in AtariPreprocessing step

# Preprocesses the environment to skip four frames, reduce screen_size to 84, initiates grey_scale
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, grayscale_newaxis=True, screen_size=84, scale_obs=False, terminal_on_life_loss=True)
env = gym.wrappers.FrameStackObservation(env, 4) # Stacks four frames, recommended in Atari RL

agent = PPOAgent(actor='highest_model_morning.pth')

def evaluate ():
    
    reward_tracker = []
    
    for _ in range(500):
        state = env.reset()
        done = False
        cumulative_reward = 0
        
        while not done:
            
            state_t = agent.state_manipulation(state)
            action, prob = agent.get_action(state_t)
            action_tensor = torch.tensor(action, dtype=torch.int64, device=device) # Convert the action to a tensor
            new_state, reward, done, trunc, info = env.step(action)
            state = new_state
            cumulative_reward += reward
            
        reward_tracker.append(cumulative_reward)
    
    df = pd.DataFrame(reward_tracker)
    df.to_excel('/mnt/rewards.xlsx')
    
