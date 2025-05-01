# DQN & Double DQN implementation with target network, epsilon decay, and stable replay buffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s_next, done):
        data = (s, a, r, s_next, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(s_next).to(device),
            torch.FloatTensor(done).to(device)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, double=False):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.double = double

        self.q_net = QNetwork(obs_dim, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
        self.total_steps = 0

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

        self.last_obs = observation
        self.last_action = action
        return [action]

    def begin_episode(self, observation):
        return self.select_action(observation)

    def step(self, reward, observation):
        self._store_transition(self.last_obs, self.last_action, reward, observation, False)
        self.train()
        return self.select_action(observation)

    def end_episode(self, reward, observation=None):
        self._store_transition(self.last_obs, self.last_action, reward, observation, True)
        self.train()

    def _store_transition(self, s, a, r, s_next, done):
        s = self.flatten_doc_obs(s['doc'])
        s_next = self.flatten_doc_obs(s_next['doc'])
        self.replay_buffer.push(s, a, r, s_next, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)

        q_vals = self.q_net(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                next_actions = self.q_net(s_next).argmax(1)
                q_next = self.q_target(s_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                q_next = self.q_target(s_next).max(1)[0]

            q_target = r + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_val, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target net
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def bundle(self):
        return {
            'q_net': self.q_net.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def unbundle(self, data):
        if data is None:
            return False
        self.q_net.load_state_dict(data['q_net'])
        self.q_target.load_state_dict(data['q_target'])
        self.optimizer.load_state_dict(data['optimizer'])
        return True
