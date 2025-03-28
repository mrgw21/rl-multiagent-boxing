import gym
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
import tensorflow as tf

def preprocess(obs):
    return tf.image.rgb_to_grayscale(obs)[..., 0:1].numpy()

def main():
    print("Listing all environments with 'Boxing' in the ID:")
    envs = gym.envs.registry.keys()
    print([env_id for env_id in envs if "Boxing" in env_id])

    env = gym.make("ALE/Boxing-v5", render_mode="human")
    if hasattr(env, "env"):
        env = env.env

    obs_shape = preprocess(env.reset()[0]).shape

    # Load agents
    ppo_agent = PPOAgent(obs_shape, env.action_space)
    ppo_agent.load("models/ppo_model.h5")

    dqn_agent = DQNAgent(obs_shape, env.action_space)
    dqn_agent.load("models/dqn_model.h5")

    obs = env.reset()[0]
    done = False
    total_reward_ppo = 0
    total_reward_dqn = 0

    while not done:
        obs_proc = preprocess(obs)

        action_ppo = ppo_agent.act(obs_proc)
        action_dqn = dqn_agent.act(obs_proc)

        # Pick one to act this frame, or alternate
        action = action_ppo  # or action_dqn or mix

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward_ppo += reward if action == action_ppo else 0
        total_reward_dqn += reward if action == action_dqn else 0

    print(f"PPO Total Reward: {total_reward_ppo}")
    print(f"DQN Total Reward: {total_reward_dqn}")
    env.close()

if __name__ == "__main__":
    main()