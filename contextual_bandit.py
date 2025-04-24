import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Linear UCB

class DocOnlyContextualBanditAgent:
    def __init__(self, doc_dim, n_arms, epsilon=0.1, lambda_reg=1):
        self.doc_dim = doc_dim
        self.n_arms = n_arms
        self.epsilon = epsilon

        #TODO: add theta
        self.lambda_reg = lambda_reg

        self.A = [np.identity(doc_dim) * lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(doc_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(doc_dim) for _ in range(n_arms)]
        # self.counts = np.zeros(n_arms)
        # self.values = np.zeros(n_arms)
        self.last_action = None
        self.last_context = None

    def begin_episode(self, observation):
        return [self.select_action(observation)]

    def step(self, reward, observation):
        self.update(self.last_action, reward)
        return [self.select_action(observation)]

    def end_episode(self, reward, observation=None):
        self.update(self.last_action, reward)

    def select_action(self, observation):
        doc_obs = observation['doc']
        context_vectors = [doc_obs[k] for k in sorted(doc_obs.keys(), key=int)]

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_arms)
        else:
            # action = np.argmax(self.values)
            scores = [self.theta[a] @ context_vectors[a] for a in range(self.n_arms)]
            action = np.argmax(scores)  

        self.last_action = action
        self.last_context = context_vectors[action]
        return action

    # def update(self, action, reward):
    #     self.counts[action] += 1
    #     n = self.counts[action]
    #     value = self.values[action]
    #     self.values[action] += (reward - value) / n

    def update(self, action, reward):
        x = self.last_context  # 当前 doc 的特征向量
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        self.theta[action] = np.linalg.inv(self.A[action]) @ self.b[action]

    def bundle(self):
        return {
            'A': self.A,
            'b': self.b,
            'theta': self.theta
    }

    def unbundle(self, data):
        if data is None:
            return False
        self.A = data['A']
        self.b = data['b']
        self.theta = data['theta']
        return True
    # def bundle(self):
    #     return {
    #         'counts': self.counts,
    #         'values': self.values
    #     }

    # def unbundle(self, data):
    #     if data is None:
    #         return False
    #     self.counts = data['counts']
    #     self.values = data['values']
    #     return True


class UserDocContextualBanditAgent:
    def __init__(self, user_dim, doc_dim, n_arms, epsilon=0.1, lambda_reg=1.0):
        self.user_dim = user_dim
        self.doc_dim = doc_dim
        self.full_dim = user_dim + doc_dim
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg

        self.A = np.array([np.identity(self.full_dim) * lambda_reg for _ in range(n_arms)])
        self.b = np.zeros((n_arms, self.full_dim))
        self.theta = np.zeros((n_arms, self.full_dim))

    def begin_episode(self, observation):
        return [self.select_action(observation)]

    def step(self, reward, observation):
        self.update(self.last_action, self.last_context, reward)
        return [self.select_action(observation)]

    def end_episode(self, reward, observation=None):
        self.update(self.last_action, self.last_context, reward)

    def select_action(self, observation):
        user_vec = observation['user']
        doc_obs = observation['doc']
        context_vectors = [
            np.concatenate([user_vec, doc_obs[k]]) for k in sorted(doc_obs.keys(), key=int)
        ]

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_arms)
        else:
            scores = []
            for i in range(self.n_arms):
                x = context_vectors[i]
                A_inv = np.linalg.inv(self.A[i])
                theta_i = A_inv @ self.b[i]
                self.theta[i] = theta_i
                score = np.dot(theta_i, x)
                scores.append(score)
            action = int(np.argmax(scores))

        self.last_action = action
        self.last_context = context_vectors[action]
        return action

    def update(self, action, context, reward):
        x = context.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * context

    def bundle(self):
        return {
            'A': self.A,
            'b': self.b,
            'theta': self.theta
        }

    def unbundle(self, data):
        if data is None:
            return False
        self.A = data['A']
        self.b = data['b']
        self.theta = data['theta']
        return True



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
