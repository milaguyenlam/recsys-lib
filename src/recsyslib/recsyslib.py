from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
from typing import Any, Callable, Protocol, Type, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from sklearn import svm
from joblib import dump, load

T = TypeVar("T")


class RecommenderModel:
    """Recommends a specified number of items based on a score calculated using a regression model"""

    def __init__(
        self,
        regression_model: RegressionModel,
        vectorizer: Vectorizer,
        training_data_loader: DataLoader,
        prediction_data_loader: DataLoader,
        prediction_handler: PredictionHandler,
        prediction_count: int = 10,
    ):
        self.vectorizer = vectorizer
        self.training_data_loader = training_data_loader
        self.prediction_data_loader = prediction_data_loader
        self.regression_model = regression_model
        self.prediction_handler = prediction_handler
        self.prediction_count = prediction_count

    def train(self):
        raw_data = self.training_data_loader.load_data()
        vectors = self.vectorizer.vectorize_data(raw_data)
        self.regression_model.fit(vectors)
        self.regression_model.dump()

    def recommend(self, input_data: list[dict]) -> list[dict]:
        vectors = self.vectorizer.vectorize_data(input_data)
        return self._compute_predictions(vectors)

    def recommend_preemptively(self):
        raw_data = self.prediction_data_loader.load_data()
        self.prediction_handler.handle(self.recommend(raw_data))

    def _compute_predictions(self, vectors: list[NDArray]):
        scored_vectors = map(lambda v: (v, self.regression_model.predict(v)), vectors)
        # return n highest scored vectors; could be generalized
        sorted_scored_vectors = sorted(scored_vectors, key=lambda x: x[1], reverse=True)
        selected_vectors = [
            scored_vector[0] for scored_vector in sorted_scored_vectors[: self.prediction_count]
        ]
        return self.vectorizer.interpret_vectors(selected_vectors)


class RegressionModel(ABC):
    @abstractmethod
    def fit(self, vectors: list[NDArray]):
        pass

    @abstractmethod
    def predict(self, vector: NDArray) -> float:
        """calculates a score for a vector"""

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def load(self):
        pass


class Vectorizer(ABC):
    # TODO: Implement domains resolving nominal and ordinal values
    def __init__(self) -> None:
        pass

    @abstractmethod
    def vectorize_data(self, data: list[dict]) -> list[NDArray]:
        pass

    @abstractmethod
    def interpret_vectors(self, vectors: list[NDArray]) -> list[dict]:
        pass


class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> list[dict]:
        pass


class PredictionHandler(ABC):
    @abstractmethod
    def handle(self, predictions: list[dict]):
        pass
