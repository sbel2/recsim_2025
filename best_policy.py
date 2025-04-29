import numpy as np

class BestPolicyAgent:
    def __init__(self, simulate_response):
        self.simulate_response = simulate_response
        self.last_action = None

    def begin_episode(self, observation):
        return [self.select_action(observation)]

    def step(self, reward, observation):
        return [self.select_action(observation)]

    def end_episode(self, reward, observation=None):
        pass

    def select_action(self, observation):
        user_vec = observation["user"]
        doc_obs = observation["doc"]

        best_doc = None
        best_watch_time = -1

        for doc_id in sorted(doc_obs.keys(), key=int):
            doc_vec = doc_obs[doc_id]
            clicked, watch_time = self.simulate_response(user_vec, doc_vec)

            if clicked and watch_time > best_watch_time:
                best_doc = int(doc_id)
                best_watch_time = watch_time

        self.last_action = best_doc if best_doc is not None else np.random.randint(len(doc_obs))
        return self.last_action

    def bundle(self):
        return {
            'last_action': self.last_action
        }

    def unbundle(self, data):
        if data is None:
            return False
        self.last_action = data.get('last_action', None)
        return True
