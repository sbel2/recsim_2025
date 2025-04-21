import abc
from gym import spaces
import numpy as np
from typing import Type, Any, List

class AbstractResponse(abc.ABC):
    """Abstract class to model a user response."""

    @staticmethod
    @abc.abstractmethod
    def response_space() -> spaces.Space:
        print("[user.py] AbstractResponse.response_space() called")
        pass

    @abc.abstractmethod
    def create_observation(self) -> Any:
        print("[user.py] AbstractResponse.create_observation() called")
        pass


class AbstractUserState(abc.ABC):
    """Abstract class to represent a user's state."""

    NUM_FEATURES: int = None  # Define in subclasses

    @abc.abstractmethod
    def create_observation(self) -> np.ndarray:
        print("[user.py] AbstractUserState.create_observation() called")
        pass

    @staticmethod
    @abc.abstractmethod
    def observation_space() -> spaces.Space:
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
        print("[user.py] AbstractUserSampler.sample_user() called")
        pass

    def get_user_ctor(self) -> Type[AbstractUserState]:
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
        print("[user.py] AbstractUserModel.update_state() called")
        pass

    def reset(self):
        print("[user.py] AbstractUserModel.reset() called")
        self._user_state = self._user_sampler.sample_user()

    def reset_sampler(self):
        print("[user.py] AbstractUserModel.reset_sampler() called")
        self._user_sampler.reset_sampler()

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        print("[user.py] AbstractUserModel.is_terminal() called")
        pass

    @abc.abstractmethod
    def simulate_response(self, documents: List[Any]) -> List[AbstractResponse]:
        print("[user.py] AbstractUserModel.simulate_response() called")
        pass

    def response_space(self) -> spaces.Space:
        print("[user.py] AbstractUserModel.response_space() called")
        res_space = self._response_model_ctor.response_space()
        return spaces.Tuple([res_space for _ in range(self._slate_size)])

    def get_response_model_ctor(self) -> Type[AbstractResponse]:
        print("[user.py] AbstractUserModel.get_response_model_ctor() called")
        return self._response_model_ctor

    def observation_space(self) -> spaces.Space:
        print("[user.py] AbstractUserModel.observation_space() called")
        return self._user_state.observation_space()

    def create_observation(self) -> np.ndarray:
        print("[user.py] AbstractUserModel.create_observation() called")
        return self._user_state.create_observation()
