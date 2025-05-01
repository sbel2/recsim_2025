import numpy as np
import helper.agent as agent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomAgent(agent.AbstractEpisodicRecommenderAgent):
    """An agent that recommends a random slate of documents."""

    def __init__(self, action_space, random_seed=0):
        super().__init__(action_space)
        self._rng = np.random.RandomState(random_seed)

    def begin_episode(self, observation):
        return self._sample_random_slate(observation)

    def step(self, reward, observation):
        return self._sample_random_slate(observation)

    def end_episode(self, reward, observation=None):
        pass  # Random agent does not learn

    def _sample_random_slate(self, observation):
        doc_obs = observation['doc']
        doc_ids = list(range(len(doc_obs)))
        self._rng.shuffle(doc_ids)
        slate = doc_ids[:self._slate_size]
        return slate

    def bundle(self):
        """Returns a checkpoint dictionary. No state to save for RandomAgent."""
        return None

    def unbundle(self, bundle):
        """Restores agent from a checkpoint. Nothing to restore."""
        return True
