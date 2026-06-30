from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(df, feature, y, preds):
    """
    Compute precision/recall/fbeta for every unique value of a categorical
    feature, used to check the model for performance disparities on slices
    of the data.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the (unprocessed) feature column. Must align
        row-for-row with y and preds (e.g. test.reset_index(drop=True)).
    feature : str
        Name of the categorical feature to slice on.
    y : np.ndarray
        Ground truth labels (binarized), aligned with df.
    preds : np.ndarray
        Model predictions (binarized), aligned with df.

    Returns
    -------
    list[dict]
        One entry per category of `feature` with precision/recall/fbeta and
        the slice size (n).
    """
    results = []
    for category in sorted(df[feature].unique()):
        mask = (df[feature] == category).values
        if mask.sum() == 0:
            continue
        precision, recall, fbeta = compute_model_metrics(y[mask], preds[mask])
        results.append(
            {
                "feature": feature,
                "category": category,
                "n": int(mask.sum()),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return results
