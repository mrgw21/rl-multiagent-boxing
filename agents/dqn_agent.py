import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(84, 84, 3)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space.n, activation='linear')
        ])
        return model

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        q_values = self.model(obs)
        return np.argmax(q_values[0].numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        minibatch = np.random.choice(self.memory, batch_size)

        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])

        # Q-value target update
        target_q = self.model(states)
        target_next_q = self.target_model(next_states)
        
        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + 0.99 * np.max(target_next_q[i])

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            loss = tf.reduce_mean(tf.square(target_q - q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
