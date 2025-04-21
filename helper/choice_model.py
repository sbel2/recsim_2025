"""
choice_model.py

Modern, PyTorch-compatible abstraction for user choice models,
converted from RecSim's original TensorFlow-based version.
"""

import abc
import numpy as np


def softmax(vector):
    """Numerically stable softmax."""
    normalized_vector = np.array(vector) - np.max(vector)
    e_x = np.exp(normalized_vector)
    return e_x / e_x.sum()


class AbstractChoiceModel(abc.ABC):
    """Abstract base class for user choice models."""

    def __init__(self):
        self._scores = None
        self._score_no_click = None

    @abc.abstractmethod
    def score_documents(self, user_state, doc_obs):
        """Compute scores for a slate of documents given the user state."""
        pass

    @abc.abstractmethod
    def choose_item(self):
        """Return the index of the chosen document, or None for no-click."""
        pass

    @property
    def scores(self):
        return self._scores

    @property
    def score_no_click(self):
        return self._score_no_click


class NormalizableChoiceModel(AbstractChoiceModel):
    """Choice models where document scores can be normalized into probabilities."""

    @staticmethod
    def _score_documents_helper(user_state, doc_obs):
        return np.array([user_state.score_document(doc) for doc in doc_obs])

    def choose_item(self):
        all_scores = np.append(self._scores, self._score_no_click)
        probabilities = all_scores / np.sum(all_scores)
        selected_index = np.random.choice(len(probabilities), p=probabilities)
        return None if selected_index == len(probabilities) - 1 else selected_index


# class MultinomialLogitChoiceModel(NormalizableChoiceModel):
#     """Choice model using softmax over raw scores + no-click mass."""

#     def __init__(self, choice_features):
#         super().__init__()
#         self._no_click_mass = choice_features.get("no_click_mass", -float("inf"))

#     def score_documents(self, user_state, doc_obs):
#         logits = self._score_documents_helper(user_state, doc_obs)
#         logits = np.append(logits, self._no_click_mass)
#         all_scores = softmax(logits)
#         self._scores = all_scores[:-1]
#         self._score_no_click = all_scores[-1]


class MultinomialProportionalChoiceModel(NormalizableChoiceModel):
    """Choice model where scores are shifted to positive range for probability."""

    def __init__(self, choice_features):
        super().__init__()
        self._min_normalizer = choice_features.get("min_normalizer", 0.0)
        self._no_click_mass = choice_features.get("no_click_mass", 0.0)

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        all_scores = np.append(scores, self._no_click_mass)
        all_scores -= self._min_normalizer
        if np.any(all_scores < 0.0):
            raise ValueError("Normalized scores have non-positive elements.")
        self._scores = all_scores[:-1]
        self._score_no_click = all_scores[-1]


# class CascadeChoiceModel(NormalizableChoiceModel):
#     """Base class for cascade-style click models (sequential scanning)."""

#     def __init__(self, choice_features):
#         super().__init__()
#         self._attention_prob = choice_features.get("attention_prob", 1.0)
#         self._score_scaling = choice_features.get("score_scaling", 1.0)

#         if not (0.0 <= self._attention_prob <= 1.0):
#             raise ValueError("attention_prob must be in [0,1].")
#         if self._score_scaling < 0.0:
#             raise ValueError("score_scaling must be positive.")

#     def _positional_normalization(self, scores):
#         self._score_no_click = 1.0
#         for i in range(len(scores)):
#             scaled = self._score_scaling * scores[i]
#             if scaled > 1.0:
#                 raise ValueError(f"Scaled score {scaled} exceeds probability bounds.")
#             scores[i] = self._score_no_click * self._attention_prob * scaled
#             self._score_no_click *= (1.0 - self._attention_prob * scaled)
#         self._scores = scores


# class ExponentialCascadeChoiceModel(CascadeChoiceModel):
#     """Cascade model with exponential scoring."""

#     def score_documents(self, user_state, doc_obs):
#         scores = self._score_documents_helper(user_state, doc_obs)
#         scores = np.exp(scores)
#         self._positional_normalization(scores)


# class ProportionalCascadeChoiceModel(CascadeChoiceModel):
#     """Cascade model with proportional scoring and min-normalization."""

#     def __init__(self, choice_features):
#         self._min_normalizer = choice_features.get("min_normalizer", 0.0)
#         super().__init__(choice_features)

#     def score_documents(self, user_state, doc_obs):
#         scores = self._score_documents_helper(user_state, doc_obs)
#         scores -= self._min_normalizer
#         if np.any(scores < 0.0):
#             raise ValueError("Normalized scores have non-positive elements.")
#         self._positional_normalization(scores)