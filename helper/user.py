import abc
from gymnasium import spaces
import numpy as np
from typing import Type, Any, List

class AbstractResponse(abc.ABC):
    """Abstract class to model a user response."""
    @staticmethod
    @abc.abstractmethod
    def response_space() -> spaces.Space:
        """Defines how a single response is represented."""
        pass

    @abc.abstractmethod
    def create_observation(self) -> Any:
        """Creates an observation of this response."""
        pass

class AbstractUserState(abc.ABC):
    """Abstract class to represent a user's state."""

    NUM_FEATURES: int = None  # Define in subclasses

    @abc.abstractmethod
    def create_observation(self) -> np.ndarray:
        """Generates observation of the user's state."""
        pass

    @staticmethod
    @abc.abstractmethod
    def observation_space() -> spaces.Space:
        """Defines how user states are represented."""
        pass

class AbstractUserSampler(abc.ABC):
    """Abstract class to sample users."""

    def __init__(self, user_ctor: Type[AbstractUserState], seed: int = 0):
        self._user_ctor = user_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def sample_user(self) -> AbstractUserState:
        """Creates a new user state instance."""
        pass

    def get_user_ctor(self) -> Type[AbstractUserState]:
        """Returns the constructor for user states."""
        return self._user_ctor

class AbstractUserModel(abc.ABC):
    """Abstract class to represent a user's dynamics."""

    def __init__(self, response_model_ctor: Type[AbstractResponse], user_sampler: AbstractUserSampler, slate_size: int):
        if not response_model_ctor:
            raise TypeError("response_model_ctor is a required callable")

        self._response_model_ctor = response_model_ctor
        self._user_sampler = user_sampler
        self._slate_size = slate_size
        self._user_state = self._user_sampler.sample_user()

    @abc.abstractmethod
    def update_state(self, slate_documents: List[Any], responses: List[AbstractResponse]):
        """Updates the user's state based on the slate and responses."""
        pass

    def reset(self):
        """Resets the user state."""
        self._user_state = self._user_sampler.sample_user()

    def reset_sampler(self):
        """Resets the user sampler."""
        self._user_sampler.reset_sampler()

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """Indicates whether the session is over."""
        pass

    @abc.abstractmethod
    def simulate_response(self, documents: List[Any]) -> List[AbstractResponse]:
        """Simulates the user's response to a slate of documents."""
        pass

    def response_space(self) -> spaces.Space:
        res_space = self._response_model_ctor.response_space()
        return spaces.Tuple([res_space for _ in range(self._slate_size)])

    def get_response_model_ctor(self) -> Type[AbstractResponse]:
        """Returns the constructor for response models."""
        return self._response_model_ctor

    def observation_space(self) -> spaces.Space:
        """Describes possible user observations."""
        return self._user_state.observation_space()

    def create_observation(self) -> np.ndarray:
        """Emits observation about the user's state."""
        return self._user_state.create_observation()