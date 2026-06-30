# Script to train machine learning model.
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, compute_slice_metrics, train_model

# Resolve paths relative to this file so the script works regardless of the
# current working directory it is invoked from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "census_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
SLICE_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "slice_output.txt")

# Add the necessary imports for the starter code.
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall precision: {precision:.4f}")
print(f"Overall recall:    {recall:.4f}")
print(f"Overall fbeta:     {fbeta:.4f}")

# Compute and save performance on slices of each categorical feature.
test_reset = test.reset_index(drop=True)
with open(SLICE_OUTPUT_PATH, "w") as f:
    f.write(f"Overall precision: {precision:.4f}\n")
    f.write(f"Overall recall:    {recall:.4f}\n")
    f.write(f"Overall fbeta:     {fbeta:.4f}\n\n")
    for feature in cat_features:
        f.write(f"=== Performance on slices of feature: {feature} ===\n")
        for row in compute_slice_metrics(test_reset, feature, y_test, preds):
            f.write(
                f"  {row['category']!s:<25} n={row['n']:<5} "
                f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
                f"fbeta={row['fbeta']:.4f}\n"
            )
        f.write("\n")
print(f"Slice performance written to {SLICE_OUTPUT_PATH}")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)
with open(os.path.join(MODEL_DIR, "lb.pkl"), "wb") as f:
    pickle.dump(lb, f)
print(f"Saved model, encoder, and label binarizer to {MODEL_DIR}")
