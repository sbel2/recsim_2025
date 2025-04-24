import numpy as np
import helper.agent as agent

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomAgent(agent.AbstractEpisodicRecommenderAgent):
    """An agent that recommends a random slate of documents."""

    def __init__(self, action_space, random_seed=0):
        # print("[Agent Init] Action Space: ", action_space)
        # [Agent Init] Action Space:  MultiDiscrete([10 10])
        # print("[Agent Init] Slate size: ", action_space.shape) #2 
        # super(RandomAgent, self).__init__(action_space)
        super().__init__(action_space)
        self._rng = np.random.RandomState(random_seed)

    def begin_episode(self, observation):
        # print("\n[Begin Episode] Observation Keys:", observation.keys())
        # Observation Keys: dict_keys(['user', 'doc', 'response'])
        # if 'user' in observation:
        #     print("[Begin Episode] User State:", observation['user'])
        #     # [1,7] vectort
        # if 'doc' in observation:
        #     print(f"[Begin Episode] Number of Docs: {len(observation['doc'])}")
        #     doc_obs = observation['doc']
        #     if isinstance(doc_obs, dict):
        #         first_key = next(iter(doc_obs))
        #         print(f"[Begin Episode] One Doc Example (truncated): {doc_obs[first_key]}")
        #     else:
        #         print(f"[Begin Episode] Unknown doc format: {type(doc_obs)}")
        #         # [1*7] vector
        # if 'response' in observation:
        #     print("[Begin Episode] Response State:", observation['response'])
        # break
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
