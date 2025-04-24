import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from naive_dqn import NaiveDQNAgent,QNetwork


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiStepDQNAgent(NaiveDQNAgent):
    def __init__(self, obs_dim, n_actions, n_step=3, epsilon=0.1, gamma=0.99, lr=1e-3):
        super().__init__(obs_dim, n_actions, epsilon, gamma, lr)
        self.n_step = n_step
        self.n_step_buffer = []

    def store_transition(self, s, a, r, s_next, done):
        if isinstance(s, dict):
            s = self.flatten_doc_obs(s['doc'])
        if isinstance(s_next, dict):
            s_next = self.flatten_doc_obs(s_next['doc'])
        self.n_step_buffer.append((s, a, r, s_next, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        R = sum([self.gamma ** i * self.n_step_buffer[i][2] for i in range(self.n_step)])
        s_0, a_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, s_n, done_n = self.n_step_buffer[-1]

        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((s_0, a_0, R, s_n, done_n))

        self.n_step_buffer.pop(0)

    def end_episode(self, reward, observation=None):
        if observation is not None:
            self.store_transition(self.prev_obs, self.last_action, reward, observation, True)
        while len(self.n_step_buffer) > 0:
            self.store_transition(*self.n_step_buffer[0])
            self.n_step_buffer.pop(0)
        self.train()
