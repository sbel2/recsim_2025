import abc


class AbstractRecommenderAgent(abc.ABC):
    """Abstract class to model a recommender system agent."""
    _multi_user = False

    def __init__(self, action_space):
        self._slate_size = action_space.nvec.shape[0]

    @property
    def multi_user(self):
        return self._multi_user

    @abc.abstractmethod
    def step(self, reward, observation):
        """Records the last transition and returns the agent's next action."""
        pass

    @abc.abstractmethod
    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a dictionary bundle of the agent's state for checkpointing."""
        pass

    @abc.abstractmethod
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        """Restores the agent's state from a checkpoint bundle."""
        pass


class AbstractEpisodicRecommenderAgent(AbstractRecommenderAgent):
    """Abstract class for episodic agents."""

    def __init__(self, action_space, summary_writer=None):
        super().__init__(action_space)
        self._episode_num = 0
        self._summary_writer = summary_writer

    def begin_episode(self, observation=None):
        self._episode_num += 1
        return self.step(0, observation)

    def end_episode(self, reward, observation=None):
        pass

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        del checkpoint_dir, iteration_number
        return {"episode_num": self._episode_num}

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        del checkpoint_dir, iteration_number
        if "episode_num" not in bundle_dict:
            print("[WARNING] Could not restore agent state from checkpoint.")
            return False
        self._episode_num = bundle_dict["episode_num"]
        return True


class AbstractMultiUserEpisodicRecommenderAgent(AbstractEpisodicRecommenderAgent):
    """Abstract agent class for multi-user recommendation environments."""
    _multi_user = True

    def __init__(self, action_space):
        self._num_users = len(action_space)
        if self._num_users <= 0:
            raise ValueError("Multi-user agent must have at least 1 user.")
        super().__init__(action_space[0])


class AbstractHierarchicalAgentLayer(AbstractRecommenderAgent):
    """Parent class for stackable (hierarchical) agent layers."""

    def __init__(self, action_space, *base_agent_ctors):
        super().__init__(action_space)
        self._base_agent_ctors = base_agent_ctors
        self._base_agents = None

    def _preprocess_reward_observation(self, reward, observation):
        """Override to add custom features or regularization."""
        return reward, observation

    @abc.abstractmethod
    def _postprocess_actions(self, action_list):
        """Combine base agent actions into final slate."""
        pass

    def begin_episode(self, observation=None):
        if observation is not None:
            _, observation = self._preprocess_reward_observation(0, observation)
        action_list = [
            agent.begin_episode(observation=observation)
            for agent in self._base_agents
        ]
        return self._postprocess_actions(action_list)

    def end_episode(self, reward, observation):
        reward, observation = self._preprocess_reward_observation(reward, observation)
        action_list = [
            agent.end_episode(reward, observation=observation)
            for agent in self._base_agents
        ]
        return self._postprocess_actions(action_list)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        bundle_dict = {}
        for i, agent in enumerate(self._base_agents):
            bundle = agent.bundle_and_checkpoint(checkpoint_dir, iteration_number)
            bundle_dict[f"base_agent_bundle_{i}"] = bundle
        return bundle_dict

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        for i, agent in enumerate(self._base_agents):
            key = f"base_agent_bundle_{i}"
            if key not in bundle_dict:
                print(f"[WARNING] Missing bundle for base agent {i}")
                return False
            if not agent.unbundle(checkpoint_dir, iteration_number, bundle_dict[key]):
                return False
        return True
