import abc


class AbstractRecommenderAgent(abc.ABC):
    """Abstract class to model a recommender system agent."""
    _multi_user = False

    def __init__(self, action_space):
        print("[agent.py] Initializing AbstractRecommenderAgent")
        self._slate_size = action_space.nvec.shape[0]

    @property
    def multi_user(self):
        return self._multi_user

    @abc.abstractmethod
    def step(self, reward, observation):
        print("[agent.py] Abstract step() called")
        pass

    @abc.abstractmethod
    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        print("[agent.py] Abstract bundle_and_checkpoint() called")
        pass

    @abc.abstractmethod
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        print("[agent.py] Abstract unbundle() called")
        pass


class AbstractEpisodicRecommenderAgent(AbstractRecommenderAgent):
    """Abstract class for episodic agents."""

    def __init__(self, action_space, summary_writer=None):
        print("[agent.py] Initializing AbstractEpisodicRecommenderAgent")
        super().__init__(action_space)
        self._episode_num = 0
        self._summary_writer = summary_writer

    def begin_episode(self, observation=None):
        print(f"[agent.py] begin_episode() - Episode {self._episode_num + 1}")
        self._episode_num += 1
        return self.step(0, observation)

    def end_episode(self, reward, observation=None):
        print(f"[agent.py] end_episode() - Episode {self._episode_num}")

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        print(f"[agent.py] bundle_and_checkpoint() - Iteration {iteration_number}")
        del checkpoint_dir, iteration_number
        return {"episode_num": self._episode_num}

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        print(f"[agent.py] unbundle() - Iteration {iteration_number}")
        del checkpoint_dir, iteration_number
        if "episode_num" not in bundle_dict:
            print("[agent.py] [WARNING] Could not restore agent state from checkpoint.")
            return False
        self._episode_num = bundle_dict["episode_num"]
        return True
