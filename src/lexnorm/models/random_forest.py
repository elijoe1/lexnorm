import os

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from lexnorm.data.normEval import loadNormData, evaluate
from lexnorm.definitions import DATA_PATH
from lexnorm.models.normalise import prep_train, prep_test, normalise


def train(train_X, train_y, output_file):
    rf_clf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight="balanced",
        oob_score=True,
    )
    rf_clf.fit(train_X, train_y)
    dump(rf_clf, output_file)


def predict(model, features, data):
    probs = model.predict_proba(features)
    data["probs"] = probs[:, 1]
    preds = data.sort_values("probs", ascending=False).drop_duplicates(
        ["process", "tweet", "tok"]
    )
    pred_tokens = preds.sort_values(["process", "tweet", "tok"]).index.tolist()
    return pred_tokens


if __name__ == "__main__":
    # train(
    #     *prep_train(os.path.join(DATA_PATH, "hpc/train_annotated.txt")),
    #     os.path.join(DATA_PATH, "../models/rf.joblib"),
    # )
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    clf = load(os.path.join(DATA_PATH, "../models/rf.joblib"))
    dev_X = prep_test(os.path.join(DATA_PATH, "hpc/dev_unannotated.txt"))
    dev = pd.read_csv(os.path.join(DATA_PATH, "hpc/dev_unannotated.txt"), index_col=0)
    pred_tokens = predict(clf, dev_X, dev)
    predictions = normalise(raw, pred_tokens)
    evaluate(raw, norm, predictions)
