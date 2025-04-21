# environment.py

import abc
import collections
import itertools

from helper.document import CandidateSet


class AbstractEnvironment(abc.ABC):
    def __init__(self, user_model, document_sampler, num_candidates, slate_size, resample_documents=True):
        print("[environment.py] AbstractEnvironment.__init__() called")
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._resample_documents = resample_documents
        self._do_resample_documents()

        if slate_size > num_candidates:
            raise ValueError(f"Slate size {slate_size} cannot exceed number of candidates {num_candidates}")

    def _do_resample_documents(self):
        print("[environment.py] AbstractEnvironment._do_resample_documents() called")
        self._candidate_set = CandidateSet()
        for _ in range(self._num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())

    @abc.abstractmethod
    def reset(self):
        print("[environment.py] AbstractEnvironment.reset() called")
        pass

    @abc.abstractmethod
    def reset_sampler(self):
        print("[environment.py] AbstractEnvironment.reset_sampler() called")
        pass

    @property
    def num_candidates(self):
        print("[environment.py] AbstractEnvironment.num_candidates accessed")
        return self._num_candidates

    @property
    def slate_size(self):
        print("[environment.py] AbstractEnvironment.slate_size accessed")
        return self._slate_size

    @property
    def candidate_set(self):
        print("[environment.py] AbstractEnvironment.candidate_set accessed")
        return self._candidate_set

    @property
    def user_model(self):
        print("[environment.py] AbstractEnvironment.user_model accessed")
        return self._user_model

    @abc.abstractmethod
    def step(self, slate):
        print("[environment.py] AbstractEnvironment.step() called")
        pass


class SingleUserEnvironment(AbstractEnvironment):
    def reset(self):
        print("[environment.py] SingleUserEnvironment.reset() called")
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())
        return user_obs, self._current_documents

    def reset_sampler(self):
        print("[environment.py] SingleUserEnvironment.reset_sampler() called")
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        print(f"[environment.py] SingleUserEnvironment.step() called with slate: {slate}")
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
        print(f"[environment.py] SingleUserEnvironment.step() done: {done}")

        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())

        return user_obs, self._current_documents, responses, done


Environment = SingleUserEnvironment  # For backward compatibility


class MultiUserEnvironment(AbstractEnvironment):
    def reset(self):
        print("[environment.py] MultiUserEnvironment.reset() called")
        for user_model in self.user_model:
            user_model.reset()
        user_obs = [user_model.create_observation() for user_model in self.user_model]
        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())
        return user_obs, self._current_documents

    def reset_sampler(self):
        print("[environment.py] MultiUserEnvironment.reset_sampler() called")
        self._document_sampler.reset_sampler()
        for user_model in self.user_model:
            user_model.reset_sampler()

    @property
    def num_users(self):
        print("[environment.py] MultiUserEnvironment.num_users accessed")
        return len(self.user_model)

    def step(self, slates):
        print(f"[environment.py] MultiUserEnvironment.step() called with {len(slates)} slates")
        if len(slates) != self.num_users:
            raise ValueError(f"Expected {self.num_users} slates, got {len(slates)}")

        for i, slate in enumerate(slates):
            if len(slate) > self._slate_size:
                raise ValueError(f"Slate {i} size {len(slate)} exceeds maximum of {self._slate_size}")

        all_user_obs = []
        all_documents = []
        all_responses = []

        for user_model, slate in zip(self.user_model, slates):
            doc_ids = list(self._current_documents)
            mapped_slate = [doc_ids[i] for i in slate]
            documents = self._candidate_set.get_documents(mapped_slate)

            if user_model.is_terminal():
                responses = []
            else:
                responses = user_model.simulate_response(documents)
                user_model.update_state(documents, responses)

            all_user_obs.append(user_model.create_observation())
            all_documents.append(documents)
            all_responses.append(responses)

        def flatten(list_of_lists):
            return list(itertools.chain.from_iterable(list_of_lists))

        self._document_sampler.update_state(flatten(all_documents), flatten(all_responses))
        done = all(user_model.is_terminal() for user_model in self.user_model)
        print(f"[environment.py] MultiUserEnvironment.step() done: {done}")

        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())

        return all_user_obs, self._current_documents, all_responses, done