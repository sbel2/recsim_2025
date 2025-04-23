import abc
from gym import spaces
import numpy as np

class CandidateSet:

    def __init__(self):
        self._documents = {}

    def size(self):
        return len(self._documents)

    def get_all_documents(self):
        return list(self._documents.values())

    def get_documents(self, document_ids):
        return [self._documents[int(doc_id)] for doc_id in document_ids]

    def add_document(self, document):
        self._documents[document.doc_id()] = document

    def remove_document(self, document):
        self._documents.pop(document.doc_id(), None)

    def create_observation(self):
        return {
            str(doc_id): doc.create_observation()
            for doc_id, doc in self._documents.items()
        }

    def observation_space(self):
        return spaces.Dict({
            str(doc_id): doc.observation_space()
            for doc_id, doc in self._documents.items()
        })

class AbstractDocumentSampler(abc.ABC):
    def __init__(self, doc_ctor, seed=0):
        self._doc_ctor = doc_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def sample_document(self):
        pass

    def get_doc_ctor(self):
        return self._doc_ctor

    @property
    def num_clusters(self):
        """Returns the number of document clusters. Defaults to 0."""
        return 0

    def update_state(self, documents, responses):
        """Updates the state based on user responses. Override if needed."""
        pass

class AbstractDocument(abc.ABC):
    NUM_FEATURES = None

    def __init__(self, doc_id):
        self._doc_id = doc_id

    def doc_id(self):
        return self._doc_id

    @abc.abstractmethod
    def create_observation(self):
        pass

    @classmethod
    @abc.abstractmethod
    def observation_space(cls):
        pass