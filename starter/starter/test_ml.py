"""
Unit tests for ml/data.py and ml/model.py.

Run with (from the starter/starter directory):
    pytest test_ml.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = ["workclass", "education", "sex"]


@pytest.fixture
def sample_data():
    """A small, fixed dataframe used across multiple tests."""
    return pd.DataFrame(
        {
            "age": [25, 38, 45, 52, 30, 60],
            "workclass": ["Private", "Self-emp", "Private", "Gov", "Private", "Gov"],
            "education": ["Bachelors", "HS-grad", "Masters", "HS-grad", "Bachelors", "Masters"],
            "sex": ["Male", "Female", "Male", "Female", "Female", "Male"],
            "hours-per-week": [40, 35, 50, 40, 30, 45],
            "salary": ["<=50K", "<=50K", ">50K", "<=50K", "<=50K", ">50K"],
        }
    )


def test_process_data_training_shapes(sample_data):
    """process_data in training mode returns X/y with matching row counts
    and a fitted encoder/binarizer."""
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    assert X.shape[0] == sample_data.shape[0]
    assert y.shape[0] == sample_data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_process_data_inference_uses_fitted_encoder(sample_data):
    """process_data in inference mode reuses the fitted encoder/binarizer and
    produces the same number of columns as training."""
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        sample_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X_test.shape[1] == X_train.shape[1]
    assert np.array_equal(y_test, y_train)


def test_train_model_returns_fitted_random_forest(sample_data):
    """train_model returns a fitted RandomForestClassifier."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference_returns_correct_number_of_predictions(sample_data):
    """inference returns one prediction per input row, with binary values."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_perfect_predictions():
    """compute_model_metrics returns 1.0 for all three metrics when
    predictions exactly match ground truth."""
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
