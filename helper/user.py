import abc
from gym import spaces
import numpy as np
from typing import Type, Any, List

class AbstractResponse(abc.ABC):
    """Abstract class to model a user response."""

    @staticmethod
    @abc.abstractmethod
    def response_space() -> spaces.Space:
        """Defines how a single response is represented."""
        print("[user.py] AbstractResponse.response_space() called")
        pass

    @abc.abstractmethod
    def create_observation(self) -> Any:
        """Creates an observation of this response."""
        print("[user.py] AbstractResponse.create_observation() called")
        pass

class AbstractUserState(abc.ABC):
    """Abstract class to represent a user's state."""

    NUM_FEATURES: int = None  # Define in subclasses

    @abc.abstractmethod
    def create_observation(self) -> np.ndarray:
        """Generates observation of the user's state."""
        print("[user.py] AbstractUserState.create_observation() called")
        pass

    @staticmethod
    @abc.abstractmethod
    def observation_space() -> spaces.Space:
        """Defines how user states are represented."""
        print("[user.py] AbstractUserState.observation_space() called")
        pass

class AbstractUserSampler(abc.ABC):
    """Abstract class to sample users."""

    def __init__(self, user_ctor: Type[AbstractUserState], seed: int = 0):
        print("[user.py] AbstractUserSampler.__init__() called")
        self._user_ctor = user_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        print("[user.py] AbstractUserSampler.reset_sampler() called")
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def sample_user(self) -> AbstractUserState:
        """Creates a new user state instance."""
        print("[user.py] AbstractUserSampler.sample_user() called")
        pass

    def get_user_ctor(self) -> Type[AbstractUserState]:
        """Returns the constructor for user states."""
        print("[user.py] AbstractUserSampler.get_user_ctor() called")
        return self._user_ctor

class AbstractUserModel(abc.ABC):
    """Abstract class to represent a user's dynamics."""

    def __init__(self, response_model_ctor: Type[AbstractResponse], user_sampler: AbstractUserSampler, slate_size: int):
        print("[user.py] AbstractUserModel.__init__() called")
        if not response_model_ctor:
            raise TypeError("response_model_ctor is a required callable")

        self._response_model_ctor = response_model_ctor
        self._user_sampler = user_sampler
        self._slate_size = slate_size
        self._user_state = self._user_sampler.sample_user()

    @abc.abstractmethod
    def update_state(self, slate_documents: List[Any], responses: List[AbstractResponse]):
        """Updates the user's state based on the slate and responses."""
        print("[user.py] AbstractUserModel.update_state() called")
        pass

    def reset(self):
        """Resets the user state."""
        print("[user.py] AbstractUserModel.reset() called")
        self._user_state = self._user_sampler.sample_user()

    def reset_sampler(self):
        """Resets the user sampler."""
        print("[user.py] AbstractUserModel.reset_sampler() called")
        self._user_sampler.reset_sampler()

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """Indicates whether the session is over."""
        print("[user.py] AbstractUserModel.is_terminal() called")
        pass

    @abc.abstractmethod
    def simulate_response(self, documents: List[Any]) -> List[AbstractResponse]:
        """Simulates the user's response to a slate of documents."""
        print("[user.py] AbstractUserModel.simulate_response() called")
        pass

    def response_space(self) -> spaces.Space:
        print("[user.py] AbstractUserModel.response_space() called")
        res_space = self._response_model_ctor.response_space()
        return spaces.Tuple([res_space for _ in range(self._slate_size)])

    def get_response_model_ctor(self) -> Type[AbstractResponse]:
        """Returns the constructor for response models."""
        print("[user.py] AbstractUserModel.get_response_model_ctor() called")
        return self._response_model_ctor

    def observation_space(self) -> spaces.Space:
        """Describes possible user observations."""
        print("[user.py] AbstractUserModel.observation_space() called")
        return self._user_state.observation_space()

    def create_observation(self) -> np.ndarray:
        """Emits observation about the user's state."""
        print("[user.py] AbstractUserModel.create_observation() called")
        return self._user_state.create_observation()