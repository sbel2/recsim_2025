import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_action(self, observation=None):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] += (reward - value) / n

class BanditAgentWrapper:
    def __init__(self, n_arms):
        self.bandit = EpsilonGreedyBandit(n_arms)

    def begin_episode(self, observation):
        action = self.bandit.select_action(observation)
        return [action]  # assume slate format

    def step(self, reward, observation):
        action = self.bandit.select_action(observation)
        self.bandit.update(action, reward)
        return [action]

    def end_episode(self, reward, observation=None):
        pass  # bandit doesn't need this

    def bundle(self):
        return None

    def unbundle(self, data):
        return False