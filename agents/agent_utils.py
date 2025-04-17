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

def test_agent(agent, episodes=5, render_mode="human", difficulty=0):
    """Run and render an episode to test the agent's performance after training."""
    agent.env = gym.make("ALE/Boxing-ram-v5", obs_type="ram", render_mode=render_mode, difficulty=difficulty)
    print(f"Testing agent at difficulty level: {difficulty}")

    total_rewards = []

    # Set epsilon to 0 for greedy policy during testing
    agent.epsilon = 0

    for episode in range(episodes):
        state, _ = agent.env.reset()  # Reset the environment
        state = agent.reduced_feature_extraction(state)  # Extract features from the initial state
        total_reward = 0
        terminal = False
        truncated = False

        while not (terminal or truncated):
            action = agent.policy(state)  # Use the agent's policy (greedy with epsilon = 0)
            next_state, reward, terminal, truncated, _ = agent.env.step(action)
            next_state = agent.reduced_feature_extraction(next_state)  # Extract features from the next state
            total_reward += reward
            state = next_state  # Move to the next state

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")

    agent.env.close()  # Close the environment after testing
    return total_rewards

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
    agent1_rewards = agent1.double_learn(episodes)
    
    # Train second agent - tile coded agent
    agent2_rewards = agent2.double_learn(episodes)
    
    # Plot comparison using moving averages
    plt.figure(figsize=(12, 6))
    
    window_size = 10
    standard_avg = [np.mean(agent1_rewards[max(0, i-window_size):i+1]) for i in range(len(agent1_rewards))]
    tile_avg = [np.mean(agent2_rewards[max(0, i-window_size):i+1]) for i in range(len(agent2_rewards))]
    
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