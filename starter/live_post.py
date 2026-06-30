"""
Sends one POST request to the live, deployed API and prints the result.

Update API_URL below to point at your deployed app (e.g. on Render) before
running.

Usage:
    python live_post.py
"""
import requests

API_URL = "https://nd0821-c3-starter-code-epnw.onrender.com/predict"

sample = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

if __name__ == "__main__":
    response = requests.post(API_URL, json=sample)
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.json()}")
