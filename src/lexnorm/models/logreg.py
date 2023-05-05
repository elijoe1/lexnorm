from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logreg(parameters, random_state=None):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(verbose=True, random_state=random_state, n_jobs=-1),
            ),
        ]
    )
    pipe.set_params(**parameters)
    return pipe
