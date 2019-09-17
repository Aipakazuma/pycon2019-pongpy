import tensorflow as tf
import numpy as np


class Agent():
    def __init__(self, model, policy=None,
                 memory=None, batch_size=256):
        self.model = model
        self.policy = policy
        self.memory = memory

        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self._compile()

        self.target_model = None
        self.target_model = tf.keras.models.clone_model(self.model)

        self.batch_size = batch_size
        self.gamma = 0.99
        self.dueling_dqn = True

    def _compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss='mse')

    def action(self, q_value):
        return self.policy.select_action(q_value)

    def update(self):
        experiences = self.memory.sample(self.batch_size)
        states = np.array([e[0] for e in experiences])
        next_states = np.array([e[1] for e in experiences])

        estimateds = self.model.predict(states)
        future = self.target_model.predict(next_states)

        rewards = 0
        for i, e in enumerate(experiences):
            reward = e[3]
            rewards += reward
            if not e[4]:
                if self.dueling_dqn:
                    reward += self.gamma * future[i][np.argmax(estimateds[i])]
                else:
                    reward += self.gamma * np.max(future[i])

            estimateds[i][e[2]] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss, rewards

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())
