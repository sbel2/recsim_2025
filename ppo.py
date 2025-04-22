import torch
import torch.nn as nn
import torch.optim as optim
import helper.agent as agent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, clip_epsilon=0.2):
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def select_action(self, state):
        user_features = torch.tensor(state['user'], dtype=torch.float32)
        probs = self.policy(user_features)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs[action.item()].detach()

    def compute_returns(self, rewards, dones):
        G = 0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, old_probs, rewards, dones):
        returns = self.compute_returns(rewards, dones)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_probs = torch.stack(old_probs)

        new_probs = self.policy(states)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        old_log_probs = torch.log(old_probs)

        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - returns.mean()

        clip_adv = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        loss = -torch.min(ratios * advantages, clip_adv).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPOAgentWrapper(agent.AbstractEpisodicRecommenderAgent):
    def __init__(self, input_dim, output_dim, action_space):
        super().__init__(action_space)
        self.agent = PPOAgent(input_dim, output_dim)
        self.states = []
        self.actions = []
        self.old_probs = []
        self.rewards = []
        self.dones = []

    def _sample_slate(self, observation):
        action, prob = self.agent.select_action(observation)
        return [action], prob

    def begin_episode(self, observation):
        action, joint_prob = self._sample_slate(observation)
        self.states = [observation['user']]
        self.actions = [action[0]]
        self.old_probs = [joint_prob]
        
        return action

    def step(self, reward, observation):
        action, joint_prob = self._sample_slate(observation)
        self.states.append(observation['user'])
        self.actions.append(action[0])
        self.old_probs.append(joint_prob)
        self.rewards.append(reward)
        self.dones.append(0)
        return action

    def end_episode(self, reward, observation=None):
        # Only append final reward if not already added
        if len(self.rewards) < len(self.actions):
            self.rewards.append(reward)
            self.dones.append(1)
        else:
            self.dones[-1] = 1  # mark the last step as done

        self.agent.update(self.states, self.actions, self.old_probs, self.rewards, self.dones)

        # Optionally reset memory
        self.states = []
        self.actions = []
        self.old_probs = []
        self.rewards = []
        self.dones = []


    def bundle(self):
        """Returns a checkpoint dictionary. Currently not saving any model state."""
        return None


    def unbundle(self, checkpoint_dir, iteration_number):
        return False