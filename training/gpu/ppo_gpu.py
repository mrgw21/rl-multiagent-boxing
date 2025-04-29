"""
This file will contain the Proximal Policy Agent (PPO).
At each transition we need to collect:
- state
- action
- advantage : This will need to be computed
- rewards
- value estimates
"""

import math
import random
import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as func

# --- Import device from neural_ne for consistency everywhere ---
from neural_ne_gpu import Actor, Critic, device

class PPOAgent:
    def __init__(self, actor=None, critic=None):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsi = 0.2
        self.entropy_coef = 0.01

        # --- Use imported device for model instantiation ---
        self.actor = Actor(18).to(device)
        self.critic = Critic().to(device)

        if actor is not None:
            self.actor.load_model(actor)
        if critic is not None:
            self.critic.load_model(critic)

        self.information = {
            'state': [],
            'state_value_function': [],
            'action': [],
            'done': [],
            'log_prob_action': [],
            'reward': [],
            'cumulative_reward': 0
        }

    def updateInformation(self, state, reward, done, trunc, info, action, action_prob):
        action = torch.tensor(action, dtype=torch.int64, device=device)
        self.information['state'].append(state)
        self.information['state_value_function'].append(self.get_state_value(state))
        self.information['done'].append(done)
        self.information['action'].append(action)
        self.information['log_prob_action'].append(action_prob)
        self.information['reward'].append(reward)
        self.information['cumulative_reward'] += reward

    def reset_information(self):
        self.information = {
            'state': [],
            'state_value_function': [],
            'done': [],
            'action': [],
            'log_prob_action': [],
            'reward': [],
            'cumulative_reward': 0
        }

    def compute_gen_advantage_estimation(self):
        gae = 0
        returns = []
        rewards = self.information['reward']
        state_value = self.information['state_value_function']
        done = self.information['done']
        mask = [1 if not x else 0 for x in done]

        for i in range(len(rewards) - 2, -1, -1):
            delta = rewards[i] + (self.gamma * mask[i] * state_value[i + 1]) - state_value[i]
            gae = delta + (self.gamma * mask[i] * self.lam * gae)
            returns.insert(0, gae + state_value[i])

        adv = torch.tensor(np.array(returns) - state_value[:-1], dtype=torch.float32, device=device)
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-10)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        return returns, adv

    def clipped_surrogate_loss(self, advantage, old_probability, new_probability):
        ratio = new_probability / old_probability
        unclipped_loss = ratio * advantage
        clipped_loss = torch.clamp(ratio, 1 - self.clip_epsi, 1 + self.clip_epsi) * advantage
        return torch.minimum(unclipped_loss, clipped_loss).mean()

    def get_action(self, state, evaluate=False):
        with torch.no_grad():
            logits = self.actor(state)
            dist = Categorical(logits=logits)
            if evaluate:
                action = torch.argmax(logits, dim=1)
                log_prob = dist.log_prob(action)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def get_state_value(self, state):
        with torch.no_grad():
            value = self.critic(state)
            return torch.squeeze(value).item()

    def calculate_losses(self, surrogate_loss, entropy, returns, value_predictions):
        entropy_bonus = self.entropy_coef * entropy
        policy_loss = (-surrogate_loss + entropy_bonus).sum()
        value_loss = func.smooth_l1_loss(returns, value_predictions).sum()
        return policy_loss, value_loss

    @staticmethod
    def state_manipulation(to_grayscale, state):
        # Normalize observation and move to correct device
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1) / 255.0
        state = to_grayscale(state)
        state = state.unsqueeze(0)
        return state.to(device)

    def learn(self, batch_size=128, epochs=4):
        """Learn from collected data with minibatch SGD."""
        states = torch.stack(self.information['state']).to(device).squeeze(1)
        actions = torch.stack(self.information['action']).to(device)
        old_action_prob = torch.tensor(self.information['log_prob_action'], dtype=torch.float32).to(device)
        rewards = torch.tensor(self.information['reward'], dtype=torch.float32).to(device)
        state_value = torch.tensor(self.information['state_value_function'], dtype=torch.float32).to(device)
        done = self.information['done']

        returns, advantages = self.compute_gen_advantage_estimation()

        dataset_size = advantages.shape[0]
        
        for _ in range(epochs):
            indices = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx].to(torch.long)

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_action_prob = old_action_prob[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_state_value = state_value[batch_idx]

                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                entropy_loss = torch.mean(dist.entropy())

                surrogate_loss = self.clipped_surrogate_loss(
                    batch_advantages, batch_old_action_prob, dist.log_prob(batch_actions)
                )
                policy_loss, value_loss = self.calculate_losses(
                    surrogate_loss, entropy_loss, batch_returns, batch_state_value
                )
                total_loss = policy_loss + 0.5 * value_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.reset_information()


    def access_cumulative_reward(self):
        return self.information['cumulative_reward']

    def access_information(self):
        return self.information
