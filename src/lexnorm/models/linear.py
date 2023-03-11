import os

import joblib
import sklearn.metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.models.normalise import prep_train, load_candidates, normalise
from lexnorm.models.predict import predict_normalisations, predict_probs


def train_logreg(candidates, output_path=None):
    train_X, train_y = prep_train(candidates)
    train_X = train_X.drop(columns="orig_same_order")
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            verbose=True,
            solver="newton-cholesky",
            class_weight="balanced",
        ),
        # LogisticRegressionCV(
        #     n_jobs=-1,
        #     scoring="f1",
        #     class_weight="balanced",
        #     solver="newton-cholesky",
        #     verbose=True,
        # ),
    )
    pipe.fit(train_X, train_y)
    if output_path is not None:
        joblib.dump(pipe, output_path)
    return pipe


if __name__ == "__main__":
    cands = load_candidates(os.path.join(DATA_PATH, "hpc/train_pipeline.txt"))
    model = train_logreg(cands)
    test_cands = load_candidates(os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"))
    predictions = predict_normalisations(
        predict_probs(model, test_cands.drop(columns="orig_same_order")), threshold=0
    )
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    predictions = normalise(raw, predictions)
    evaluate_predictions(raw, norm, predictions)
