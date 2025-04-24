# environment.py

import abc
import collections
import itertools

from helper.document import CandidateSet


class AbstractEnvironment(abc.ABC):
    def __init__(self, user_model, document_sampler, num_candidates, slate_size, resample_documents=True):
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._resample_documents = resample_documents
        self._do_resample_documents()

        if slate_size > num_candidates:
            raise ValueError(f"Slate size {slate_size} cannot exceed number of candidates {num_candidates}")

    def _do_resample_documents(self):
        self._candidate_set = CandidateSet()
        for _ in range(self._num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def reset_sampler(self):
        pass

    @property
    def num_candidates(self):
        return self._num_candidates

    @property
    def slate_size(self):
        return self._slate_size

    @property
    def candidate_set(self):
        return self._candidate_set

    @property
    def user_model(self):
        return self._user_model

    @abc.abstractmethod
    def step(self, slate):
        pass


class SingleUserEnvironment(AbstractEnvironment):
    def reset(self):
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())
        return user_obs, self._current_documents

    def reset_sampler(self):
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        if len(slate) > self._slate_size:
            raise ValueError(f"Slate size {len(slate)} exceeds maximum of {self._slate_size}")

        doc_ids = list(self._current_documents)
        mapped_slate = [doc_ids[i] for i in slate]
        documents = self._candidate_set.get_documents(mapped_slate)

        responses = self._user_model.simulate_response(documents)
        self._user_model.update_state(documents, responses)
        self._document_sampler.update_state(documents, responses)

        user_obs = self._user_model.create_observation()
        done = self._user_model.is_terminal()

        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())

        return user_obs, self._current_documents, responses, done


Environment = SingleUserEnvironment  # For backward compatibility