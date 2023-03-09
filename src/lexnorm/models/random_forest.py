import os

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from lexnorm.data.baseline import mfr
from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.definitions import LEX_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.generate_extract.process import create_index
from lexnorm.models.normalise import prep_train, prep_test, normalise, load_candidates


# TODO: make train and predict able to take dataframe not just path for cross validation
def train(candidates, output_file):
    train_X, train_y = prep_train(candidates)
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        # class_weight="balanced",
        oob_score=True,
        # suggested by scikit docs as good to prevent over-fitting (seems to work a little...?)
        min_samples_leaf=5
        # prevent over-fitting
        # turns out don't need this with abstaining if low prob. Brings precision down, recall up.
        # HIGHER THAN MFR!!
        # max_depth=3,
    )
    rf_clf.fit(train_X, train_y)
    dump(rf_clf, output_file)


def predict_probs(model, candidates):
    features = prep_test(candidates)
    probs = model.predict_proba(features)
    candidates = candidates.copy()
    candidates["probs"] = probs[:, 1]
    return candidates


def predict_normalisations(dataframe, threshold=0.5):
    # 0.5 is not an arbitrary threshold - if above .predict function of classifier would predict class 1
    # as takes class with highest proba and there are only two classes. NOTE that .predict_proba should not be interpreted
    # as confidence level of class for random forest - just number input to decision function.
    # TODO: if tie for highest probability, just chooses arbitrarily
    pred_df = dataframe.sort_values("probs", ascending=False).drop_duplicates(
        ["tok_id"]
    )
    pred_df = pred_df.loc[pred_df.probs >= threshold]
    # pred_df = pred_df.sort_values(["tok_id"]).index.tolist()
    pred_df["candidate"] = pred_df.index.values
    # as values are singleton lists
    return {
        k: v[0]
        for k, v in pred_df.groupby("tok_id")["candidate"].apply(list).to_dict().items()
    }


if __name__ == "__main__":
    # TODO as random state fixed, shuffling dataset can change performance metric -
    #  perhaps cross validation comes in useful here?
    train_df = load_candidates(
        os.path.join(DATA_PATH, "hpc/train_pipeline.txt"), shuffle=True
    )
    dev_df = load_candidates(
        os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"), shuffle=True
    )
    train(
        train_df,
        os.path.join(DATA_PATH, "../models/rf.joblib"),
    )
    dev_raw, dev_norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    clf = load(os.path.join(DATA_PATH, "../models/rf.joblib"))
    pred_tokens = predict_normalisations(
        predict_probs(clf, dev_df),
        threshold=0.5,
    )
    predictions = normalise(
        dev_raw, pred_tokens, os.path.join(DATA_PATH, "../models/output.txt")
    )
    evaluate_predictions(dev_raw, dev_norm, predictions)
    train_raw, train_norm = loadNormData(os.path.join(DATA_PATH, "raw/train.norm"))
    evaluate_predictions(dev_raw, dev_norm, mfr(train_raw, train_norm, dev_raw))
