# PPO paper link: https://arxiv.org/pdf/1707.06347

import math
import random
import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as func
import math

# --- Import device from neural_ne for consistency everywhere ---
from neural_ne_gpu import Actor, Critic, device

class PPOAgent:
    def __init__(self, actor=None, critic=None):
        """
        Initialise the PPO agent.
        Args:
            actor (Actor): The actor model.
            critic (Critic): The critic model.
        """
        self.gamma = 0.99 # Discount factor
        self.lam = 0.95 # Lambda
        self.clip_epsi = 0.2 # Keep the same, as per PPO paper
        self.entropy_coef = 0.01 # Entropy coefficient

        # Use imported device for model instantiation 
        self.actor = Actor(18).to(device)
        self.critic = Critic().to(device)

        if actor is not None:
            self.actor.load_model(actor)
        if critic is not None:
            self.critic.load_model(critic)

        # Information is a dictionary that will store the data for each episode
        self.information = {
            'state': [],
            'state_value_function': [],
            'action': [],
            'done': [],
            'log_prob_action': [],
            'reward': [],
            'cumulative_reward': 0
        }
        
        self.loss_tracker = [] # Tracks the loss function

    def updateInformation(self, state, reward, done, trunc, info, action, action_prob):
        """
        Update the information dictionary.
        Args:
            state (torch.Tensor): The state.
            reward (float): The reward.
            done (bool): Whether the episode is done.
            action (torch.Tensor): The action determined by the policy.
            action_prob (torch.Tensor): Log probability of the Actor NN output distribution
        """
        
        self.information['state'].append(state)
        self.information['done'].append(done)
        self.information['action'].append(action)
        self.information['log_prob_action'].append(action_prob)
        self.information['reward'].append(reward)
        self.information['state_value_function'].append(self.get_state_value(state))
        self.information['cumulative_reward'] += reward

    def reset_information(self):
        """
        Reset the information dictionary.
        """
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
        """
        Compute the general advantage estimation (GAE).
        """
        # Get rewards, state values, and done flags
        rewards = self.information['reward']
        state_value = self.information['state_value_function']
        done = self.information['done']
        
        gae = 0 # Initial GAE of first state
        returns = [] # Storing returns
        mask = [1 if not x else 0 for x in done] # Mask is a list of 1s and 0s, where 1s are for done = False and 0s for done = True
        
        # Reverse the rewards and state values
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * state_value[i + 1] * mask[i]) - state_value[i] # delta is the difference between the reward and the state value
            gae = delta + (self.gamma * self.lam * mask[i] * gae) # gae is the advantage
            returns.insert(0, gae + state_value[i]) # Inserts at the start of the list

        returns = torch.tensor(returns, dtype=torch.float32).to(device) # Convert to tensor
        returns = (returns - returns.mean()) / (returns.std() + 1e-10) # Normalises the returns
        adv = returns - torch.tensor(state_value[:-1], dtype=torch.float32).to(device) # Computes the advantages
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-10) # Normalise the advantage
        
        return returns, adv
    
    def add_final_state_value(self, state):
        """
        Add the final state value to compute the advantage of the final state in the batch.
        Args:
            state (torch.Tensor): The state.
        """
        state_t = self.state_manipulation(state)
        self.information['state_value_function'].append(self.get_state_value(state_t)) # Append the state value to the information dictionary


    def clipped_surrogate_loss(self, advantage, old_probability, new_probability):
        """
        Calculates the clipped surrogate loss.
        Args:
            advantage (torch.Tensor): The advantages.
            old_probability (torch.Tensor): The old probabilities.
            new_probability (torch.Tensor): The new probabilities.
        Returns:
            float: The clipped surrogate loss.
        """
        ratio = torch.exp(new_probability - old_probability) # ratio is the ratio of the new probability to the old probability
        unclipped_loss = ratio * advantage # unclipped loss is the loss of the new policy
        clipped_loss = torch.clamp(ratio, 1 - self.clip_epsi, 1 + self.clip_epsi) * advantage # clipped loss is the loss of the old policy
        return -(torch.min(unclipped_loss, clipped_loss)).mean() # return the negative mean of the minimum of the unclipped and clipped loss

    def get_action(self, state, evaluate=False):
        """
        Gets the action.
        Args:
            state (torch.Tensor): The state.
            evaluate (bool): Whether to evaluate.
        Returns:
            tuple: The action and the log probability.
        """
        with torch.no_grad():
            logits = self.actor(state)
            dist = Categorical(logits=logits) # dist is the distribution of the action
            if evaluate:
                action = torch.argmax(logits, dim=1) # if evaluate, then the action is the argmax of the logits
            else:
                action = dist.sample() # if not evaluate, then the action is the sample of the distribution
            
            log_prob = torch.squeeze(dist.log_prob(action)).item() # log probability of the action
            action = torch.squeeze(action).item() # Retrieves action
            
            return action, log_prob

    def get_state_value(self, state):
        """
        Gets the state value.
        Args:
            state (torch.Tensor): The state.
        Returns:
            float: The state value.
        """
        with torch.no_grad():
            value = self.critic(state) # Pass the state through the critic NN
        return torch.squeeze(value).item()

    def calculate_losses(self, surrogate_loss, entropy, returns, old_value_prediction, value_predictions):
        """
        Calculates the losses.
        Policy loss simply calculated.
        Value loss is clipped in this function.
        
        Args:
            surrogate_loss (torch.Tensor): The surrogate loss.
            entropy (torch.Tensor): The entropy.
            returns (torch.Tensor): The returns.
            old_value_prediction (torch.Tensor): The old value prediction.
            value_predictions (torch.Tensor): The value predictions.
        Returns:
            tuple: The policy loss and the value loss.
        """
        
        # Policy loss
        entropy_bonus = self.entropy_coef * entropy # entropy bonus, to encourage exploration 
        policy_loss = surrogate_loss - entropy_bonus # final policy loss
        
        
        # Value loss
        value_pred_clipped = old_value_prediction + (value_predictions - old_value_prediction).clamp(self.clip_epsi, self.clip_epsi)
        value_loss_unclipped = (value_predictions - returns) ** 2 # value loss unclipped is the loss of the new policy
        value_loss_clipped = (value_pred_clipped - returns) ** 2 # value loss clipped is the loss of the old policy
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean() # value loss is the mean of the maximum of the value loss unclipped and value loss clipped
        
        return policy_loss, value_loss

    @staticmethod
    def state_manipulation(state):
        """
        Manipulates the state to reconfigure the height, width, channels and batch size.
        Args:
            state (torch.Tensor): The state.
        Returns:
            torch.Tensor: The manipulated state.
        """
        if isinstance(state, tuple):
            state = state[0] # Gets first element of tuple
        state = torch.tensor(state, dtype=torch.float32) / 255.0 # Normalise the observation
        state = state.permute(3, 0, 1, 2) # Permute the observation
        return state.to(device)

    def learn(self, batch_size=128, epochs = 4):
        """
        Learn from collected data with minibatch SGD.
        Args:
            batch_size (int): The batch size.
            epochs (int): The number of epochs.
        """
        states = torch.stack(self.information['state']).to(device).squeeze(1) # Stack the states
        actions = torch.stack(self.information['action']).to(device) # Stack the actions
        old_action_prob = torch.tensor(self.information['log_prob_action'], dtype=torch.float32).to(device) # Old action probability
        rewards = torch.tensor(self.information['reward'], dtype=torch.float32).to(device)
        state_values = torch.tensor(self.information['state_value_function'], dtype = torch.float32).to(device)
        done = self.information['done']

        returns, advantages = self.compute_gen_advantage_estimation() # Compute the general advantage estimation
        loss = [] 
        
        # Finds dataset size
        dataset_size = advantages.shape[0]
        
        # Four epochs on this whole batch of 100
        for _ in range(epochs):
            indices = torch.randperm(dataset_size)
            
            # In slices of 4 the agent is trained
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx].to(torch.long)

                # Retrieves information from the larger batch for learning
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_action_prob = old_action_prob[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_state_value = self.critic(batch_states).squeeze()
                old_batch_state_values = state_values[batch_idx]


                # Computes new policy
                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                entropy_loss = torch.mean(dist.entropy())

                surrogate_loss = self.clipped_surrogate_loss(
                    batch_advantages, batch_old_action_prob, dist.log_prob(batch_actions)
                ) # Computes the clipped surrogate loss 
                policy_loss, value_loss = self.calculate_losses(
                    surrogate_loss, entropy_loss, batch_returns, old_batch_state_values, batch_state_value
                ) # Computes the policy loss and the value loss

                self.actor.optimizer.zero_grad() # Zero the gradients
                policy_loss.backward() # Backpropagate the policy loss
                self.actor.optimizer.step() # Update the actor

                self.critic.optimizer.zero_grad() # Zero the gradients
                value_loss.backward() # Backpropagate the value loss
                self.critic.optimizer.step() # Update the critic
                
                total_loss = policy_loss + value_loss * 0.5 # Computes the total loss
                loss.append(total_loss)
                
        self.reset_information() # Reset the information
        final_loss = final_loss = np.array([l.detach().cpu().item() for l in loss]) 
        self.loss_tracker.append(np.mean(final_loss)) # Append the final loss to the loss tracker


    def access_cumulative_reward(self):
        """
        Accesses the cumulative reward.
        Returns:
            float: The cumulative reward.
        """
        return self.information['cumulative_reward']

    def access_information(self):
        """
        Accesses the information.
        Returns:
            dict: The information.
        """
        return self.information
