import abc
from gym import spaces
import numpy as np

class CandidateSet:
    """Represents a collection of AbstractDocuments, indexed by their document ID."""

    def __init__(self):
        print("[document.py] CandidateSet initialized")
        self._documents = {}

    def size(self):
        print("[document.py] CandidateSet.size() called")
        return len(self._documents)

    def get_all_documents(self):
        print("[document.py] CandidateSet.get_all_documents() called")
        return list(self._documents.values())

    def get_documents(self, document_ids):
        print(f"[document.py] CandidateSet.get_documents() called with IDs: {document_ids}")
        return [self._documents[int(doc_id)] for doc_id in document_ids]

    def add_document(self, document):
        print(f"[document.py] CandidateSet.add_document() called for doc_id: {document.doc_id()}")
        self._documents[document.doc_id()] = document

    def remove_document(self, document):
        print(f"[document.py] CandidateSet.remove_document() called for doc_id: {document.doc_id()}")
        self._documents.pop(document.doc_id(), None)

    def create_observation(self):
        print("[document.py] CandidateSet.create_observation() called")
        return {
            str(doc_id): doc.create_observation()
            for doc_id, doc in self._documents.items()
        }

    def observation_space(self):
        print("[document.py] CandidateSet.observation_space() called")
        return spaces.Dict({
            str(doc_id): doc.observation_space()
            for doc_id, doc in self._documents.items()
        })


class AbstractDocumentSampler(abc.ABC):
    """Abstract base class for sampling documents."""

    def __init__(self, doc_ctor, seed=0):
        print("[document.py] AbstractDocumentSampler initialized")
        self._doc_ctor = doc_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        print(f"[document.py] AbstractDocumentSampler.reset_sampler() with seed: {self._seed}")
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def sample_document(self):
        print("[document.py] Abstract sample_document() called")
        pass

    def get_doc_ctor(self):
        print("[document.py] get_doc_ctor() called")
        return self._doc_ctor

    @property
    def num_clusters(self):
        print("[document.py] num_clusters() called")
        return 0

    def update_state(self, documents, responses):
        print("[document.py] update_state() called")
        pass


class AbstractDocument(abc.ABC):
    """Abstract base class representing a document and its properties."""

    NUM_FEATURES = None  # Should be defined in subclasses

    def __init__(self, doc_id):
        print(f"[document.py] AbstractDocument initialized with doc_id: {doc_id}")
        self._doc_id = doc_id

    def doc_id(self):
        return self._doc_id

    @abc.abstractmethod
    def create_observation(self):
        print("[document.py] Abstract create_observation() called")
        pass

    @classmethod
    @abc.abstractmethod
    def observation_space(cls):
        print("[document.py] Abstract observation_space() called")
        pass
