import abc
from gym import spaces
import numpy as np

class CandidateSet:
    """Represents a collection of AbstractDocuments, indexed by their document ID."""

    def __init__(self):
        self._documents = {}

    def size(self):
        """Returns the number of documents in the candidate set."""
        return len(self._documents)

    def get_all_documents(self):
        """Retrieves all documents in the candidate set."""
        return list(self._documents.values())

    def get_documents(self, document_ids):
        """Retrieves documents corresponding to the specified document IDs.

        Args:
            document_ids: An iterable of document IDs (integers or strings).

        Returns:
            A list of AbstractDocument instances.
        """
        return [self._documents[int(doc_id)] for doc_id in document_ids]

    def add_document(self, document):
        """Adds a document to the candidate set.

        Args:
            document: An instance of AbstractDocument.
        """
        self._documents[document.doc_id()] = document

    def remove_document(self, document):
        """Removes a document from the candidate set.

        Args:
            document: An instance of AbstractDocument.
        """
        self._documents.pop(document.doc_id(), None)

    def create_observation(self):
        """Creates an observation dictionary of all documents.

        Returns:
            A dictionary mapping document IDs to their observations.
        """
        return {
            str(doc_id): doc.create_observation()
            for doc_id, doc in self._documents.items()
        }

    def observation_space(self):
        """Creates a Gym space dictionary for all documents.

        Returns:
            A gym.spaces.Dict instance representing the observation space.
        """
        return spaces.Dict({
            str(doc_id): doc.observation_space()
            for doc_id, doc in self._documents.items()
        })

class AbstractDocumentSampler(abc.ABC):
    """Abstract base class for sampling documents."""

    def __init__(self, doc_ctor, seed=0):
        self._doc_ctor = doc_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        """Resets the random number generator for sampling."""
        self._rng = np.random.default_rng(self._seed)

    @abc.abstractmethod
    def sample_document(self):
        """Samples and returns an instance of AbstractDocument."""
        pass

    def get_doc_ctor(self):
        """Returns the constructor/class of the documents to be sampled."""
        return self._doc_ctor

    @property
    def num_clusters(self):
        """Returns the number of document clusters. Defaults to 0."""
        return 0

    def update_state(self, documents, responses):
        """Updates the state based on user responses. Override if needed."""
        pass

class AbstractDocument(abc.ABC):
    """Abstract base class representing a document and its properties."""

    NUM_FEATURES = None  # Should be defined in subclasses

    def __init__(self, doc_id):
        self._doc_id = doc_id

    def doc_id(self):
        """Returns the document ID."""
        return self._doc_id

    @abc.abstractmethod
    def create_observation(self):
        """Returns observable properties of the document as a float array."""
        pass

    @classmethod
    @abc.abstractmethod
    def observation_space(cls):
        """Defines the Gym space representing the document's observation."""
        pass
