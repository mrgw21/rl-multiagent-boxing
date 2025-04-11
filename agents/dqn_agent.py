# agents/dqn_agent.py — Full PyTorch DQNAgent for Inference and Training

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, action_space, input_shape=(4, 84, 84), gamma=0.99, lr=1e-4, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, device=None):
        self.action_space = action_space
        self.input_shape = input_shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(self.input_shape, self.action_space.n).to(self.device)
        self.target_model = QNetwork(self.input_shape, self.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=2000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space.n)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return int(torch.argmax(q_values[0]).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device) / 255.0
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device) / 255.0
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
