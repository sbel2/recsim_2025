import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from naive_dqn_re import NaiveDQNAgent,QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DoubleDQNAgent(NaiveDQNAgent):
    def __init__(self, obs_dim, n_actions, epsilon=0.1, gamma=0.99, lr=1e-4, tau=0.01):
        super().__init__(obs_dim, n_actions, epsilon, gamma, lr)
        self.target_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.tau = tau  # for soft update
        # epsilon_decay allow early explore, late exploite
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.train_step = 0


    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)
        
        s = torch.FloatTensor(s).to(device)
        s_next = torch.FloatTensor(s_next).to(device)
        a = torch.LongTensor(a).to(device)
        r = torch.FloatTensor(r).to(device)
        done = torch.FloatTensor(done).to(device)

        q_vals = self.q_net(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN 核心：当前网络选 action，target 网络给 value
        with torch.no_grad():
            next_q_values = self.q_net(s_next)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            target_q_values = self.target_net(s_next)
            q_next = target_q_values.gather(1, next_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_val, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_target_network()

        # decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_target_network(self):
        # Soft update: target = tau * q_net + (1 - tau) * target_net
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def bundle(self):
        base = super().bundle()
        base['target_net_state_dict'] = self.target_net.state_dict()
        return base

    def unbundle(self, data):
        if not super().unbundle(data):
            return False
        self.target_net.load_state_dict(data['target_net_state_dict'])
        return True

