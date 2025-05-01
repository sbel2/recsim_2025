import numpy as np

class BestPolicyAgent:
    def __init__(self, doc_dim, n_arms, slate_size=2):
        self.doc_dim = doc_dim
        self.n_arms = n_arms
        self.slate_size = slate_size

    def begin_episode(self, observation):
        return self.select_action(observation)

    def step(self, reward, observation):
        return self.select_action(observation)

    def end_episode(self, reward, observation=None):
        pass

    def select_action(self, observation):
        user_vec = np.asarray(observation['user'], dtype=np.float32).flatten()

        doc_obs_raw = observation['doc']
        doc_obs = {int(k): v for k, v in doc_obs_raw.items()}
        sorted_doc_ids = sorted(doc_obs.keys())
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(sorted_doc_ids)}

        scores = []
        for doc_id in sorted_doc_ids:
            doc_vec = np.asarray(doc_obs[doc_id], dtype=np.float32).flatten()
            affinity = np.dot(user_vec, doc_vec)
            score = affinity
            scores.append((doc_id, score))

        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:self.slate_size]
        action = [doc_id_to_index[doc_id] for doc_id, _ in top_k]

        return action
    
    def bundle(self):
        return None

    def unbundle(self, data):
        return False






