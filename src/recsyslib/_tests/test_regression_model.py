from recsyslib import RecommenderModel, RegressionModel, Vectorizer
from numpy.typing import NDArray

import os
import json


def load_json_file(path):
    with open(path, "r", encoding="utf8") as json_file:
        return json.load(json_file)


TEST_VECTORS = load_json_file(f"{os.getcwd()}/_tests/test_data/test_vectors.json")
# ---------- TESTING ---------- #
def test_selection_of_predictions():
    dummy_recommender_model = get_dummy_recommender_model(*[True] * 5)
    print(dummy_recommender_model._compute_predictions(TEST_VECTORS))


def get_dummy_recommender_model(
    mock_regression_model: bool = False,
    mock_vectorizer: bool = False,
    mock_training_data_loader: bool = False,
    mock_prediction_data_loader: bool = False,
    mock_prediction_handler: bool = False,
) -> RecommenderModel:
    regression_model = DummyRegressionModel()
    vectorizer = DummyVectorizer()
    training_data_loader = DummyDataLoader()
    prediction_data_loader = DummyDataLoader()
    prediction_handler = DummyPredictionHandler()
    return RecommenderModel(
        regression_model,
        vectorizer,
        training_data_loader,
        prediction_data_loader,
        prediction_handler,
        prediction_count=3,
    )


class DummyRegressionModel(RegressionModel):
    def fit(self, vectors: list[NDArray]):
        pass

    def predict(self, vector: NDArray) -> float:
        return vector["test_score"]

    def dump(self):
        return super().dump()

    def load(self):
        return super().load()


class DummyVectorizer(Vectorizer):
    def vectorize_data(self, data: list[dict]) -> list[NDArray]:
        return data

    def interpret_vectors(self, vectors: list[NDArray]) -> list[dict]:
        return vectors


class DummyDataLoader:
    pass


class DummyPredictionHandler:
    pass


if __name__ == "__main__":
    test_selection_of_predictions()
