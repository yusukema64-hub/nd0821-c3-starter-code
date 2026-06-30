"""
FastAPI application that serves the census income classification model.
"""
import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Resolve paths relative to this file so they work both locally and once
# deployed to the cloud application platform.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)


class CensusSample(BaseModel):
    """A single row of census data used as input to the model.

    Aliases are used for fields whose original column name contains a
    hyphen, since hyphens are not valid Python identifiers.
    """

    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Married-civ-spouse")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


app = FastAPI(title="Census Income Classifier API")


@app.get("/")
async def welcome():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the Census Income Classification API!"}


@app.post("/predict")
async def predict(sample: CensusSample):
    """Run model inference on a single sample of census data.

    Returns the predicted salary class: '<=50K' or '>50K'.
    """
    data = pd.DataFrame([sample.model_dump(by_alias=True)])

    X, _, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]

    return {"prediction": label}
