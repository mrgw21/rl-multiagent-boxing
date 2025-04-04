import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, action_space, input_shape=(84, 84, 3), gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.action_space = action_space
        self.input_shape = input_shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.action_space.n, activation='linear')
        ])
        return model

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space.n)
        obs = np.expand_dims(obs, axis=0)
        q_values = self.model(obs, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        target_q = self.model(states).numpy()
        target_next_q = self.target_model(next_states).numpy()

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

        return loss.numpy()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
