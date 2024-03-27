from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from config.logging_config import info_log

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


def train_model(x_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    x_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    info_log("Training model.")

    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                  intercept_scaling=1, l1_ratio=None, max_iter=100,
                                  multi_class='ovr', n_jobs=None, penalty='l2',
                                  random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                  warm_start=False)

    lr_model.fit(x_train, y_train)

    return lr_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    info_log("Computing model metrics.")

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    info_log("Running inference.")

    return model.predict(X)
