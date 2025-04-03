import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from training.metrics import MetricsLogger

class DQNAgent:
    def __init__(self, action_space, state_shape):
        self.action_space = action_space.n
        self.state_shape = state_shape
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = []
        self.max_memory_size = 10000
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)(inputs)
        x = layers.Conv2D(32, 8, strides=4, activation='relu')(x)
        x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
        x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(self.action_space, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Dummy loss

        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, targets = [], []

        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                q_next = np.max(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                target[action] = reward + self.gamma * q_next
            states.append(state)
            targets.append(target)

        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(history.history["loss"][0])


def preprocess(obs):
    return np.mean(obs, axis=2).astype(np.uint8)[::2, ::2, np.newaxis]  # grayscale + downsample


def train_dqn_agent(episodes=1000, save_path="models/dqn_model.keras", metrics_dir="output"):
    env = gym.make("ALE/Boxing-v5")
    if hasattr(env, 'env'):
        env = env.env

    obs = env.reset()
    obs = preprocess(obs)
    agent = DQNAgent(env.action_space, obs.shape)
    logger = MetricsLogger(save_path=metrics_dir, run_name="dqn")

    for episode in range(episodes):
        obs = preprocess(env.reset())
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            next_obs = preprocess(next_obs)

            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train()

            obs = next_obs
            total_reward += reward
            steps += 1

        logger.log(episode, total_reward, steps, loss)
        print(f"Episode {episode + 1}: Return = {total_reward}, Steps = {steps}, Loss = {loss:.4f}")

        if (episode + 1) % 20 == 0:
            agent.model.save(save_path)
            print(f"📦 Saved model to {save_path}")

    logger.save()
    logger.plot()
    env.close()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    train_dqn_agent()
