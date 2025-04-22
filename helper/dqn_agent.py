import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ResponseAdapter:
#     def __init__(self, input_response_space):
#         self._input_response_space = input_response_space
#         self._single_response_space = input_response_space.spaces[0]
#         self._response_names = list(self._single_response_space.spaces.keys())
#         self._response_shape = (len(input_response_space.spaces),
#                                 len(self._response_names))
#         self._response_dtype = np.float32

#     @property
#     def response_names(self):
#         return self._response_names

#     @property
#     def response_shape(self):
#         return self._response_shape

#     @property
#     def response_dtype(self):
#         return self._response_dtype

#     def encode(self, responses):
#         tensor = np.zeros(self._response_shape, dtype=self._response_dtype)
#         for i, response in enumerate(responses):
#             for j, key in enumerate(self.response_names):
#                 tensor[i, j] = response[key]
#         return tensor


class ObservationAdapter:
    def __init__(self, input_observation_space, stack_size=1):
        self._input_observation_space = input_observation_space
        self._user_space = input_observation_space.spaces['user']
        self._doc_space = input_observation_space.spaces['doc']
        self._num_candidates = len(self._doc_space.spaces)

        user_dim = spaces.flatdim(self._user_space)
        doc_dim = spaces.flatdim(list(self._doc_space.spaces.values())[0])
        self._feature_dim = max(user_dim, doc_dim)

        self._observation_shape = (self._num_candidates + 1, self._feature_dim)
        self._observation_dtype = np.float32
        self._stack_size = stack_size

    def _pad_with_zeros(self, array):
        pad_width = self._feature_dim - len(array)
        return np.pad(array, (0, pad_width), mode='constant')

    @property
    def output_observation_space(self):
        low = np.full(self._observation_shape, -np.inf, dtype=self._observation_dtype)
        high = np.full(self._observation_shape, np.inf, dtype=self._observation_dtype)
        return spaces.Box(low=low, high=high, dtype=self._observation_dtype)

    def encode(self, observation):
        image = np.zeros(self._observation_shape, dtype=self._observation_dtype)

        try:
            user_flat = spaces.flatten(self._user_space, observation['user'])
        except Exception as e:
            print("[ERROR] Failed to flatten user observation!")
            print(f"[ERROR] observation['user']: {observation['user']}")
            raise e

        # Encode user
        user_features = self._pad_with_zeros(spaces.flatten(self._user_space, observation['user']))
        image[0] = user_features

        for i, (space, doc_obs) in enumerate(zip(self._doc_space.spaces.values(), observation['doc'].values())):
            try:
                doc_features = self._pad_with_zeros(spaces.flatten(space, doc_obs))
            except Exception as e:
                print(f"[ERROR] Failed to flatten doc {i}!")
                raise e

            if i + 1 >= self._observation_shape[0]:
                break  # prevent overflow
            image[i + 1] = doc_features
        return image


class RecSimDQN(nn.Module):
    def __init__(self, user_dim, doc_dim, num_docs, num_actions):
        super().__init__()
        self.user_fc = nn.Linear(user_dim, 64)
        self.docs_fc = nn.Linear(doc_dim * num_docs, 128)
        self.combine_fc = nn.Linear(64 + 128, 64)
        self.q = nn.Linear(64, num_actions)

    def forward(self, x):
        # x shape: (batch_size, num_docs + 1, feature_dim)
        user = x[:, 0, :]  # (batch_size, feature_dim)
        docs = x[:, 1:, :].reshape(x.size(0), -1)  # flatten docs

        user_feat = torch.relu(self.user_fc(user))
        docs_feat = torch.relu(self.docs_fc(docs))
        combined = torch.cat([user_feat, docs_feat], dim=1)
        hidden = torch.relu(self.combine_fc(combined))
        return self.q(hidden)

# Lightweight Replay Buffer (cyclic)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgentRecSim:
    def __init__(self, observation_space, num_actions, stack_size=1,
                 optimizer_name='adam', eval_mode=False, summary_writer=None, **kwargs):

        self.eval_mode = eval_mode
        self._obs_adapter = ObservationAdapter(observation_space, stack_size)
        self._observation_shape = self._obs_adapter.output_observation_space.shape
        self._observation_dtype = self._obs_adapter.output_observation_space.dtype
        self._summary_writer = summary_writer
        self.num_actions = num_actions

        user_dim = spaces.flatdim(observation_space.spaces['user'])
        doc_dim = spaces.flatdim(list(observation_space.spaces['doc'].spaces.values())[0])
        num_docs = len(observation_space.spaces['doc'].spaces)

        self.q_network = RecSimDQN(user_dim, doc_dim, num_docs, num_actions)
        self.target_network = RecSimDQN(user_dim, doc_dim, num_docs, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = self._get_optimizer(optimizer_name)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.total_steps = 0

        self.last_observation = None
        self.last_action = None

    def _get_optimizer(self, name):
        if name == 'adam':
            return optim.Adam(self.q_network.parameters(), lr=1e-3)
        elif name == 'sgd':
            return optim.SGD(self.q_network.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def begin_episode(self, observation):
        encoded_obs = self._obs_adapter.encode(observation)
        self.last_observation = encoded_obs
        self.last_action = self._select_action(encoded_obs)
        return self.last_action

    def step(self, reward, observation):
        encoded_obs = self._obs_adapter.encode(observation)
        self.replay_buffer.push(self.last_observation, self.last_action, reward, encoded_obs, False)

        if len(self.replay_buffer) >= self.batch_size and not self.eval_mode:
            self._learn_from_batch()

        self.last_observation = encoded_obs
        self.last_action = self._select_action(encoded_obs)
        return self.last_action

    def end_episode(self, reward, observation=None):
        if observation is not None:
            encoded_obs = self._obs_adapter.encode(observation)
        else:
            encoded_obs = np.zeros_like(self.last_observation)
        self.replay_buffer.push(self.last_observation, self.last_action, reward, encoded_obs, True)

    def _select_action(self, encoded_obs):
        if not self.eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        obs_tensor = torch.FloatTensor(encoded_obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax(dim=1).item()

    def _learn_from_batch(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach().unsqueeze(1)
        targets = rewards + self.gamma * next_q_values * (~dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def bundle(self):
        return {
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.total_steps
        }

    def unbundle(self, bundle):
        if 'model_state_dict' not in bundle:
            print("[WARNING] Missing model state for unbundling.")
            return False
        self.q_network.load_state_dict(bundle['model_state_dict'])
        self.target_network.load_state_dict(bundle['target_state_dict'])
        self.optimizer.load_state_dict(bundle['optimizer_state_dict'])
        self.epsilon = bundle.get('epsilon', 1.0)
        self.total_steps = bundle.get('steps', 0)
        return True