import collections
import numpy as np
from gym import Env, spaces
from helper.environment import MultiUserEnvironment


def _dummy_metrics_aggregator(responses, metrics, info):
    print("[rec_gym.py] _dummy_metrics_aggregator() called")
    return metrics


def _dummy_metrics_writer(metrics, add_summary_fn):
    print("[rec_gym.py] _dummy_metrics_writer() called")
    pass


class RecSimGymEnv(Env):
    """Gym-compliant wrapper for a recommendation simulation environment."""

    def __init__(
        self,
        raw_environment,
        reward_aggregator,
        metrics_aggregator=_dummy_metrics_aggregator,
        metrics_writer=_dummy_metrics_writer,
    ):
        print("[rec_gym.py] RecSimGymEnv.__init__() called")
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer
        self.reset_metrics()

    @property
    def environment(self):
        print("[rec_gym.py] environment property accessed")
        return self._environment

    @property
    def game_over(self):
        print("[rec_gym.py] game_over property accessed")
        return False

    @property
    def action_space(self):
        print("[rec_gym.py] action_space property accessed")
        base_space = spaces.MultiDiscrete(
            self._environment.num_candidates * np.ones((self._environment.slate_size,), dtype=int)
        )
        if isinstance(self._environment, MultiUserEnvironment):
            return spaces.Tuple([base_space] * self._environment.num_users)
        return base_space

    @property
    def observation_space(self):
        print("[rec_gym.py] observation_space property accessed")
        if isinstance(self._environment, MultiUserEnvironment):
            user_obs_space = self._environment.user_model[0].observation_space()
            response_obs_space = self._environment.user_model[0].response_space()
            user_obs_space = spaces.Tuple([user_obs_space] * self._environment.num_users)
            response_obs_space = spaces.Tuple([response_obs_space] * self._environment.num_users)
        else:
            user_obs_space = self._environment.user_model.observation_space()
            response_obs_space = self._environment.user_model.response_space()

        return spaces.Dict({
            "user": user_obs_space,
            "doc": self._environment.candidate_set.observation_space(),
            "response": response_obs_space,
        })

    def step(self, action):
        print(f"[rec_gym.py] step() called with action: {action}")
        user_obs, doc_obs, responses, done = self._environment.step(action)

        if isinstance(self._environment, MultiUserEnvironment):
            all_responses = tuple(
                tuple(resp.create_observation() for resp in user_resps)
                for user_resps in responses
            )
        else:
            all_responses = tuple(resp.create_observation() for resp in responses)

        obs = {
            "user": user_obs,
            "doc": doc_obs,
            "response": all_responses
        }

        reward = self._reward_aggregator(responses)
        info = self.extract_env_info()

        print(f"[rec_gym.py] step() returning reward: {reward}, done: {done}")
        return obs, reward, done, info

    def reset(self):
        print("[rec_gym.py] reset() called")
        user_obs, doc_obs = self._environment.reset()
        return {
            "user": user_obs,
            "doc": doc_obs,
            "response": None
        }

    def reset_sampler(self):
        print("[rec_gym.py] reset_sampler() called")
        self._environment.reset_sampler()

    def render(self, mode='human'):
        print("[rec_gym.py] render() called")
        raise NotImplementedError("Render not implemented.")

    def close(self):
        print("[rec_gym.py] close() called")
        pass

    def seed(self, seed=None):
        print(f"[rec_gym.py] seed() called with seed: {seed}")
        np.random.seed(seed)

    def extract_env_info(self):
        print("[rec_gym.py] extract_env_info() called")
        return {"env": self._environment}

    def reset_metrics(self):
        print("[rec_gym.py] reset_metrics() called")
        self._metrics = collections.defaultdict(float)

    def update_metrics(self, responses, info=None):
        print("[rec_gym.py] update_metrics() called")
        self._metrics = self._metrics_aggregator(responses, self._metrics, info)

    def write_metrics(self, add_summary_fn):
        print("[rec_gym.py] write_metrics() called")
        self._metrics_writer(self._metrics, add_summary_fn)
