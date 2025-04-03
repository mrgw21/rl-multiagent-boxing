import os
import random
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, optimizers
from datetime import datetime
from training.metrics import MetricsLogger

# Setup
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# TensorFlow multi-GPU setup
gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()

print("Num GPUs Available:", len(gpus))
print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Hyperparameters
GAMMA = 0.99
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
TOTAL_EPISODES = 1
METRICS_PATH = "metrics/ppo_metrics.csv"
MODEL_PATH = "models/ppo_model.h5"

class PPOAgent:
    def __init__(self, obs_shape, action_space):
        with strategy.scope():
            self.actor = self.build_actor(obs_shape, action_space)
            self.critic = self.build_critic(obs_shape)
            self.actor_old = self.build_actor(obs_shape, action_space)
            self.actor_old.set_weights(self.actor.get_weights())
            self.actor_optimizer = optimizers.Adam(ACTOR_LR)
            self.critic_optimizer = optimizers.Adam(CRITIC_LR)
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    def build_actor(self, obs_shape, action_space):
        inputs = layers.Input(shape=obs_shape)
        x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
        x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(action_space.n, activation='softmax')(x)
        return Model(inputs, outputs)

    def build_critic(self, obs_shape):
        inputs = layers.Input(shape=obs_shape)
        x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
        x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return Model(inputs, outputs)

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0).astype(np.float32) / 255.0
        probs = self.actor(obs).numpy()[0]
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]

    def train(self, observations, actions, advantages, old_probs, returns):
        obs = np.array(observations, dtype=np.float32) / 255.0
        actions = np.array(actions)
        advantages = np.array(advantages, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_batches = 0

        for _ in range(UPDATE_EPOCHS):
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            for start in range(0, len(obs), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idxs = idxs[start:end]

                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    logits = self.actor(obs[batch_idxs])
                    values = self.critic(obs[batch_idxs])[:, 0]
                    new_probs = tf.reduce_sum(logits * tf.one_hot(actions[batch_idxs], self.actor.output_shape[-1]), axis=1)
                    ratio = new_probs / (old_probs[batch_idxs] + 1e-10)
                    clip_adv = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages[batch_idxs]
                    loss_actor = -tf.reduce_mean(tf.minimum(ratio * advantages[batch_idxs], clip_adv))
                    entropy = -tf.reduce_mean(tf.reduce_sum(logits * tf.math.log(logits + 1e-10), axis=1))
                    loss_critic = tf.reduce_mean((returns[batch_idxs] - values) ** 2)
                    total_loss_actor = loss_actor - ENTROPY_COEF * entropy

                grads_actor = tape1.gradient(total_loss_actor, self.actor.trainable_variables)
                grads_critic = tape2.gradient(loss_critic, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

                total_actor_loss += loss_actor.numpy()
                total_critic_loss += loss_critic.numpy()
                num_batches += 1

        self.actor_old.set_weights(self.actor.get_weights())
        self.last_actor_loss = total_actor_loss / max(1, num_batches)
        self.last_critic_loss = total_critic_loss / max(1, num_batches)

def preprocess(obs):
    return tf.image.rgb_to_grayscale(obs)[..., 0:1].numpy()

def compute_advantages(rewards, values, gamma=GAMMA):
    returns, G = [], 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    advantages = returns - np.array(values)
    return advantages, returns

def main():
    env = gym.make("ALE/Boxing-v5", render_mode=None)
    obs_shape = preprocess(env.reset()[0]).shape
    agent = PPOAgent(obs_shape, env.action_space)
    logger = MetricsLogger(save_path="metrics", run_name="ppo")  # Save CSV to metrics/

    for episode in range(TOTAL_EPISODES):
        print(f"--- Episode {episode} ---")
        obs, done, steps = env.reset()[0], False, 0
        obs_list, action_list, prob_list, reward_list, value_list = [], [], [], [], []

        while not done:
            obs_proc = preprocess(obs)
            action, prob = agent.get_action(obs_proc)
            value = agent.critic(np.expand_dims(obs_proc / 255.0, axis=0)).numpy()[0, 0]

            obs_list.append(obs_proc)
            action_list.append(action)
            prob_list.append(prob)
            value_list.append(value)

            obs, reward, done, _, _ = env.step(action)
            reward_list.append(reward)
            steps += 1

        advs, rets = compute_advantages(reward_list, value_list)
        agent.train(obs_list, action_list, advs, prob_list, rets)

        logger.log(episode, sum(reward_list), steps, agent.last_actor_loss + agent.last_critic_loss)
        print(f"Episode {episode} completed: reward={sum(reward_list)}, steps={steps}")

    logger.save()
    logger.plot()
    plt.savefig("plots/ppo_plot.png")
    agent.actor.save(MODEL_PATH)
    print(f"Training complete. Outputs saved to:\n  - {METRICS_PATH}\n  - {MODEL_PATH}\n  - plots/ppo_plot.png")



if __name__ == "__main__":
    main()
