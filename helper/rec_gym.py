import collections
import numpy as np
from gym import Env, spaces


def _dummy_metrics_aggregator(responses, metrics, info):
    return metrics


def _dummy_metrics_writer(metrics, add_summary_fn):
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
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer
        self.reset_metrics()

    @property
    def environment(self):
        return self._environment

    @property
    def game_over(self):
        return False

    @property
    def action_space(self):
        """Returns action space: slate of document indices."""
        base_space = spaces.MultiDiscrete(
            self._environment.num_candidates * np.ones((self._environment.slate_size,), dtype=int)
        )
        return base_space

    @property
    def observation_space(self):
        """Returns observation space with user, document, and response info."""
        user_obs_space = self._environment.user_model.observation_space()
        response_obs_space = self._environment.user_model.response_space()

        return spaces.Dict({
            "user": user_obs_space,
            "doc": self._environment.candidate_set.observation_space(),
            "response": response_obs_space,
        })

    def step(self, action):
        """Executes one step of the environment."""
        user_obs, doc_obs, responses, done = self._environment.step(action)
        all_responses = tuple(resp.create_observation() for resp in responses)

        obs = {
            "user": user_obs,
            "doc": doc_obs,
            "response": all_responses
        }

        reward = self._reward_aggregator(responses)
        info = self.extract_env_info()

        return obs, reward, done, info

    def reset(self):
        """Resets the environment."""
        user_obs, doc_obs = self._environment.reset()
        return {
            "user": user_obs,
            "doc": doc_obs,
            "response": None
        }

    def reset_sampler(self):
        self._environment.reset_sampler()

    def render(self, mode='human'):
        raise NotImplementedError("Render not implemented.")

    def seed(self, seed=None):
        np.random.seed(seed)

    def extract_env_info(self):
        return {"env": self._environment}

    def reset_metrics(self):
        self._metrics = collections.defaultdict(float)

    def update_metrics(self, responses, info=None):
        self._metrics = self._metrics_aggregator(responses, self._metrics, info)

    def write_metrics(self, add_summary_fn):
        self._metrics_writer(self._metrics, add_summary_fn)