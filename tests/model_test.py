from ml.model import train_model, compute_model_metrics, inference
from sklearn.linear_model import LogisticRegression
import pytest
import numpy as np


# Sample data for testing
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
y_preds = np.random.randint(0, 2, size=100)


@pytest.fixture
def trained_model():
    return train_model(X_train, y_train)


def test_train_model_returns_model():
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics_returns_floats():
    precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference_returns_numpy_array(trained_model):
    preds = inference(trained_model, X_train)
    assert isinstance(preds, np.ndarray)
