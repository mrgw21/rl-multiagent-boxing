import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def save_agent(agent, path):
    """Save an agent as a pickled object."""
    with open(path, 'wb') as file_:
        pickle.dump(agent, file_)
    print(f"Agent successfully saved to {path}.")

def load_agent(path):
    """Load an agent from a pickled object."""
    with open(path, 'rb') as file_:
        agent = pickle.load(file_)
    print(f"Agent successfully loaded from {path}.")
    return agent

def test_agent(agent, render_mode="human"):
    """Run and render an episode to test the agent's performance after training."""
    env = gym.make(agent.env.spec.id, render_mode=render_mode)  # Recreate the environment with rendering
    state, _ = env.reset()  # Unpack the reset tuple
    total_reward = 0
    terminal = False
    truncated = False

    while not (terminal or truncated):
        state_features = agent.extract_state_feature(state)  # Use the agent's method to extract features
        action = agent.policy(state_features, greedy=True)  # Use greedy policy for testing
        state, reward, terminal, truncated, _ = env.step(action)
        total_reward += reward

    env.close()  # Close the environment after testing
    print(f"Total Reward: {total_reward}")
    return total_reward

def plot_learning_curve(rewards):
    plt.figure(figsize=(12, 6))

    # Plot individual episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.grid(True)

    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = 10
    moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel(f"{window_size}-Episode Moving Average")
    plt.title("Moving Average of Rewards")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_agents(agent1, agent2, episodes=100):
    """
    Function that compares agents training performance.
    Runs training for agents and then plots comparison on a graph.
    Returns both agents so they can be saved.
    """
    # Train first agent - in this case, the standard agent
    standard_rewards = agent1.double_learn(episodes)
    
    # Train second agent - tile coded agent
    tile_rewards = agent2.double_learn(episodes)
    
    # Plot comparison using moving averages
    plt.figure(figsize=(12, 6))
    
    window_size = 10
    standard_avg = [np.mean(standard_rewards[max(0, i-window_size):i+1]) for i in range(len(standard_rewards))]
    tile_avg = [np.mean(tile_rewards[max(0, i-window_size):i+1]) for i in range(len(tile_rewards))]
    
    plt.plot(standard_avg, label='Standard Agent')
    plt.plot(tile_avg, label='Tile Coded Agent')
    plt.xlabel("Episode")
    plt.ylabel(f"{window_size}-Episode Moving Average Reward")
    plt.title("Standard vs Tile Coded Agent Performance")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return both agents
    return agent1, agent2