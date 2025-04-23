import os
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, graph_name="Learning Curve", save_path=None):
        """
        Function to plot and save learning curves
        """
        def moving_average(data, window_size=500):
            # Just performs moving average calculation to help with interpretability 
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(rewards), linewidth=2, color='red', label='Smoothed')
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
        plt.title(f'Learning Curve {graph_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def save_agent(agent, name):
    """Save an agent as a pickled object."""
    path = f"saved_agents/testing_agents/{name}.pkl"
    with open(path, 'wb') as file_:
        pickle.dump(agent, file_)
    print(f"Agent successfully saved to {path}.")

def load_agent(name, best=False):
    """Load an agent from a pickled object."""
    # Used for the agents in the best agents folder
    if best:
        path = f"saved_agents/best_agents/{name}.pkl"
    else:
        path = f"saved_agents/testing_agents/{name}.pkl"
    with open(path, 'rb') as file_:
        agent = pickle.load(file_)
    print(f"Agent successfully loaded from {path}.")
    return agent

def test_agent(agent, episodes=5, render_mode="human", difficulty=3, cartpole=False):
    """Run and render an episode to test the agent's performance after training."""
    if cartpole:
        env = gym.make('CartPole-v1')
    else:
        env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render_mode, difficulty=difficulty) # Removed difficulty = difficulty
    print(f"Testing Agent with difficulty: {difficulty}")
    agent.env = env
    # Set epsilon to 0 for greedy policy during testing
    agent.epsilon = 0
    
    # print(f"Testing agent at difficulty level: {difficulty}")
    total_rewards = []

    for episode in range(episodes):
        state, _ = agent.env.reset()  # Reset the environment
        state = agent.feature_extraction(state)
        total_reward = 0
        terminal = False
        truncated = False

        while not (terminal or truncated):
            action = agent.policy(state)  # Use the agent's policy (greedy with epsilon = 0)
            next_state, reward, terminal, truncated, _ = agent.env.step(action)
            next_state = agent.feature_extraction(next_state)
            total_reward += reward
            state = next_state  # Move to the next state

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")

    env.close()  # Close the environment after testing
    return total_rewards
