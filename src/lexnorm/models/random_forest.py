import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from lexnorm.data.baseline import mfr
from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.definitions import LEX_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.generate_extract.process import create_index
from lexnorm.models.normalise import prep_train, prep_test, normalise, load_candidates


# TODO: make train and predict able to take dataframe not just path for cross validation
def train(candidates, random_state, output_file):
    train_X, train_y = prep_train(candidates)
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
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


def train_predict_evaluate_cv(
    model_dir, tweets_path, df_dir, output_dir=None, train_first=False
):
    raw, norm = loadNormData(tweets_path)
    raw = np.array(raw, dtype=object)
    norm = np.array(norm, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # to make robust to randomness of classifier while giving consistent results
    model_rng = np.random.RandomState(42)
    load_rng = np.random.RandomState(42)
    comb_raw = []
    comb_norm = []
    comb_preds = []
    mfr_preds = []
    for i, folds in enumerate(kf.split(raw, norm)):
        train_idx, test_idx = folds
        test_raw = raw[test_idx].tolist()
        test_norm = norm[test_idx].tolist()
        comb_raw += test_raw
        comb_norm += test_norm
        train_raw = raw[train_idx].tolist()
        train_norm = norm[train_idx].tolist()
        mfr_preds += mfr(train_raw, train_norm, test_raw)
        train_df = create_index(
            load_candidates(
                os.path.join(DATA_PATH, df_dir, f"train_{i}.txt"),
                random_state=load_rng,
                shuffle=True,
            )
        )
        dev_df = create_index(
            load_candidates(
                os.path.join(DATA_PATH, df_dir, f"test_{i}.txt"),
                random_state=load_rng,
                shuffle=True,
            )
        )
        if train_first:
            train(
                train_df,
                model_rng,
                os.path.join(DATA_PATH, model_dir, f"rf_{i}.joblib"),
            )
        clf = load(os.path.join(DATA_PATH, model_dir, f"rf_{i}.joblib"))
        pred_tokens = predict_normalisations(
            predict_probs(clf, dev_df),
            threshold=0.5,
        )
        comb_preds += normalise(
            test_raw,
            pred_tokens,
            os.path.join(DATA_PATH, output_dir, f"output_{i}.txt")
            if output_dir is not None
            else None,
        )
    evaluate_predictions(comb_raw, comb_norm, comb_preds)
    evaluate_predictions(comb_raw, comb_norm, mfr_preds)


def train_predict_evaluate(
    model_path,
    train_tweets_path,
    test_tweets_path,
    train_df_path,
    test_df_path,
    output_path=None,
    train_first=False,
):
    model_rng = np.random.RandomState(42)
    load_rng = np.random.RandomState(42)
    train_df = load_candidates(train_df_path, random_state=load_rng, shuffle=True)
    test_df = load_candidates(test_df_path, random_state=load_rng, shuffle=True)
    raw, norm = loadNormData(test_tweets_path)
    if train_first:
        train(train_df, model_rng, model_path)
    clf = load(model_path)
    pred_tokens = predict_normalisations(
        predict_probs(clf, test_df),
        threshold=0.5,
    )
    predictions = normalise(raw, pred_tokens, output_path)
    evaluate_predictions(raw, norm, predictions)
    train_raw, train_norm = loadNormData(train_tweets_path)
    evaluate_predictions(raw, norm, mfr(train_raw, train_norm, raw))


if __name__ == "__main__":
    train_predict_evaluate(
        os.path.join(DATA_PATH, "../models/rf.joblib"),
        os.path.join(DATA_PATH, "raw/train.norm"),
        os.path.join(DATA_PATH, "raw/dev.norm"),
        os.path.join(DATA_PATH, "hpc/train_pipeline.txt"),
        os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"),
        os.path.join(DATA_PATH, "../models/output.txt"),
    )
    # train_predict_evaluate_cv(
    #     os.path.join(DATA_PATH, "../models"),
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "hpc/cv"),
    #     os.path.join(DATA_PATH, "../models/output"),
    # )
