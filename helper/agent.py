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