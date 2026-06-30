"""
Unit tests for the FastAPI application (starter/main.py).

Run with (from the starter/ directory):
    pytest test_main.py
"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    """GET on the root should return 200 and the welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Classification API!"}


def test_post_predict_low_income():
    """POST /predict on a profile expected to be classified '<=50K'."""
    sample = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "9th",
        "education-num": 5,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 10,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert body["prediction"] == "<=50K"


def test_post_predict_high_income():
    """POST /predict on a profile expected to be classified '>50K'."""
    sample = {
        "age": 45,
        "workclass": "Self-emp-inc",
        "fnlgt": 200000,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 90,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert body["prediction"] == ">50K"
