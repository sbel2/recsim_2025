import numpy as np

class UCBBandit:
    def __init__(self, n_arms, c=1.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0

    def select_action(self, observation=None):
        self.total_pulls += 1

        ucb_scores = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
            avg_reward = self.values[arm]
            confidence = self.c * np.sqrt(np.log(self.total_pulls) / self.counts[arm])
            ucb_scores[arm] = avg_reward + confidence

        return int(np.argmax(ucb_scores))

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] += (reward - value) / n

class BanditAgentWrapper:
    def __init__(self, n_arms, c=1.0):
        self.bandit = UCBBandit(n_arms, c)
        self.last_action = None

    def begin_episode(self, observation):
        self.last_action = self.bandit.select_action(observation)
        return [self.last_action]

    def step(self, reward, observation):
        self.bandit.update(self.last_action, reward)
        self.last_action = self.bandit.select_action(observation)
        return [self.last_action]

    def end_episode(self, reward, observation=None):
        self.bandit.update(self.last_action, reward)

    def bundle(self):
        return None

    def unbundle(self, data):
        return False
