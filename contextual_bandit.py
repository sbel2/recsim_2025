import numpy as np

class ContextualBanditAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def begin_episode(self, observation):
        action = self.select_action(observation)
        return [action]

    def step(self, reward, observation):
        self.update(self.last_action, reward)
        action = self.select_action(observation)
        return [action]

    def end_episode(self, reward, observation=None):
        self.update(self.last_action, reward)

    def select_action(self, observation=None):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_arms)
        else:
            action = np.argmax(self.values)
        self.last_action = action
        return action

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] += (reward - value) / n

    def bundle(self):
        return {
            'counts': self.counts,
            'values': self.values
        }

    def unbundle(self, data):
        if data is None:
            return False
        self.counts = data['counts']
        self.values = data['values']
        return True
