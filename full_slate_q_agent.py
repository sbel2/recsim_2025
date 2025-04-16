import itertools
from gym import spaces
from helper import agent as abstract_agent
from helper import dqn_agent

class FullSlateQAgent(dqn_agent.DQNAgentRecSim, abstract_agent.AbstractEpisodicRecommenderAgent):
    """Recommender agent implementing full slate Q-learning."""

    def __init__(
        self,
        observation_space,
        action_space,
        optimizer_name='adam',
        eval_mode=False,
        **kwargs
    ):
        self._num_candidates = int(action_space.nvec[0])
        abstract_agent.AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._all_possible_slates = list(itertools.permutations(
            range(self._num_candidates),
            action_space.nvec.shape[0]
        ))

        self._env_action_space = spaces.Discrete(len(self._all_possible_slates))

        dqn_agent.DQNAgentRecSim.__init__(
            self,
            observation_space=observation_space,
            num_actions=len(self._all_possible_slates),
            stack_size=1,
            optimizer_name=optimizer_name,
            eval_mode=eval_mode,
            **kwargs
        )

    def begin_episode(self, observation):
        return self._all_possible_slates[super().begin_episode(observation)]

    def step(self, reward, observation):
        return self._all_possible_slates[super().step(reward, observation)]

    def end_episode(self, reward, observation):
        super().end_episode(reward, observation)

