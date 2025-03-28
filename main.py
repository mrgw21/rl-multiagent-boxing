import gym
from agents.random_agent import RandomAgent
import numpy as np

# Patch for numpy.bool8 issue (optional, based on numpy version)
np.bool8 = bool

def main():
    print("Listing all environments with 'Boxing' in the ID:")
    envs = gym.envs.registry.keys()
    print([env_id for env_id in envs if "Boxing" in env_id])

    # Create environment
    env = gym.make("ALE/Boxing-v5", render_mode="human")

    # Unwrap TimeLimit to avoid 5-value unpacking
    if hasattr(env, "env"):
        env = env.env

    obs = env.reset()
    agent = RandomAgent(env.action_space)

    done = False
    total_reward = 0

    while not done:
        action = agent.act(obs)
        # Correctly unpacking the 5 values returned by step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
