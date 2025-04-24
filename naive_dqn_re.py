import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(QNetwork, self).__init__()
        # TODO: 加深网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class NaiveDQN_target_Agent:
    def __init__(self, obs_dim, n_actions, epsilon=1, gamma=0.99, lr=1e-4):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        ## TODO: add epsilon_decay
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.target_net = deepcopy(self.q_net).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []  # 简单 replay buffer
        self.batch_size = 32
        self.max_memory = 10000
        self.train_step = 0
        self.target_update_freq = 10

    def begin_episode(self, observation):
        self.prev_obs = observation
        action = self.select_action(observation)
        return [action]

    def step(self, reward, observation):
        self.store_transition(self.prev_obs, self.last_action, reward, observation, False)
        self.train()
        self.prev_obs = observation
        action = self.select_action(observation)
        return [action]

    def end_episode(self, reward, observation=None):
        self.store_transition(self.prev_obs, self.last_action, reward, observation, True)
        self.train()

    def flatten_doc_obs(self, doc_obs_dict):
        keys = sorted(doc_obs_dict.keys(), key=int)
        flat = np.concatenate([doc_obs_dict[k] for k in keys])
        return flat.astype(np.float32)

    def select_action(self, observation):
        flat_obs = self.flatten_doc_obs(observation['doc'])

        obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(device)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)
                action = q_values.argmax().item()
        self.last_action = action
        return action

    def store_transition(self, s, a, r, s_next, done):
        s = self.flatten_doc_obs(s['doc'])
        s_next = self.flatten_doc_obs(s_next['doc'])
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.FloatTensor(s).to(device)
        a = torch.LongTensor(a).to(device)
        r = torch.FloatTensor(r).to(device)
        s_next = torch.FloatTensor(s_next).to(device)
        done = torch.FloatTensor(done).to(device)

        q_vals = self.q_net(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_net(s_next).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - done)
        loss = self.loss_fn(q_val, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: 同步target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def bundle(self):
        return {
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def unbundle(self, data):
        if data is None:
            return False
        self.q_net.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.target_net.load_state_dict(data['model_state_dict'])
        return True
