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
from lexnorm.models.normalise import prep_train, prep_test, normalise

# TODO: make train and predict able to take dataframe not just path for cross validation
def train(data_path, output_file):
    data = pd.read_csv(
        data_path, index_col=0, keep_default_na=False, na_values=""
    ).sample(
        frac=1,
        # random_state=42,
    )
    train_X, train_y = prep_train(data)
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight="balanced",
        oob_score=True,
        # prevent over-fitting
        max_depth=5,
    )
    rf_clf.fit(train_X, train_y)
    dump(rf_clf, output_file)


def predict_probs(model, data_path):
    data = pd.read_csv(
        data_path, index_col=0, keep_default_na=False, na_values=""
    ).sample(
        frac=1,
        # random_state=42,
    )
    features = prep_test(data)
    probs = model.predict_proba(features)
    data = data.copy()
    data["probs"] = probs[:, 1]
    return data


def predict_normalisations(dataframe, threshold=0.5):
    # TODO: if tie for highest probability, just chooses arbitrarily
    dataframe = create_index(dataframe)
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
    # TODO as random state fixed, shuffling dataset can hugely change performance metric -
    #  perhaps cross validation comes in useful here?
    # train(
    #     os.path.join(DATA_PATH, "hpc/train_ngrams.txt"),
    #     os.path.join(DATA_PATH, "../models/rf.joblib"),
    # )
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    clf = load(os.path.join(DATA_PATH, "../models/rf.joblib"))
    pred_tokens = predict_normalisations(
        predict_probs(clf, os.path.join(DATA_PATH, "hpc/dev_ngrams.txt")), threshold=0.5
    )
    predictions = normalise(
        raw, pred_tokens, os.path.join(DATA_PATH, "../models/output.txt")
    )
    evaluate_predictions(raw, norm, predictions)
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/train.norm"))
    dev_raw, dev_norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    evaluate_predictions(dev_raw, dev_norm, mfr(raw, norm, dev_raw))
