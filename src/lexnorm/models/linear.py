import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.models.normalise import load_candidates, normalise
from lexnorm.models.predict import predict_normalisations, predict_probs


def create_logreg(parameters, random_state=None):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    verbose=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.set_params(**parameters)
    return pipe
