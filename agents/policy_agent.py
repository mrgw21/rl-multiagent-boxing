import tensorflow as tf
import numpy as np

class PolicyAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = self.create_model()
        self.critic_model = self.create_critic_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(84, 84, 3)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space.n, activation='softmax')
        ])
        return model

    def create_critic_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(84, 84, 3)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        policy = self.model(obs)
        action = np.random.choice(self.action_space.n, p=policy[0].numpy())
        return action

    def learn(self, states, actions, rewards, next_states, done):
        with tf.GradientTape() as tape:
            # Value function (critic) prediction
            values = self.critic_model(states)
            next_values = self.critic_model(next_states)
            
            # Advantage estimation (TD-error)
            td_target = rewards + (1 - done) * 0.99 * next_values
            advantages = td_target - values
            
            # Policy loss (Actor)
            action_probs = self.model(states)
            action_log_probs = tf.math.log(action_probs)
            selected_action_log_probs = tf.reduce_sum(action_log_probs * tf.one_hot(actions, self.action_space.n), axis=1)
            ratio = tf.exp(selected_action_log_probs - tf.stop_gradient(action_log_probs))
            
            # PPO objective (Clipping)
            clip_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_ratio * advantages))
            
            # Critic loss (Value function)
            critic_loss = tf.reduce_mean(tf.square(advantages))

            loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(loss, self.model.trainable_variables + self.critic_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + self.critic_model.trainable_variables))

        return loss.numpy()