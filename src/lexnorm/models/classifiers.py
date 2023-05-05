import os

import numpy as np
from joblib import dump, load
from sklearn.model_selection import KFold
from sklearn.tree import plot_tree

from lexnorm.data.baseline import mfr
from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.generate_extract.candidates import create_index
from lexnorm.models.logreg import create_logreg

from lexnorm.models.normalise import prep_train, load_candidates
from lexnorm.models.predict import (
    predict_candidates,
    predict_probs,
    predict_normalisation,
)
from lexnorm.models.random_forest import create_rf


def train_model(model, candidates, output_path=None):
    train_X, train_y = prep_train(candidates)
    model.fit(train_X, train_y)
    if output_path is not None:
        with open(output_path, "wb") as f:
            dump(model, f)
    return model


def train_predict_evaluate_cv(
    model,
    model_dir,
    tweets_path,
    df_dir,
    output_dir=None,
    train_first=False,
    drop_features=None,
    with_mfr=False,
):
    raw, norm = loadNormData(tweets_path)
    raw = np.array(raw, dtype=object)
    norm = np.array(norm, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # to make robust to randomness of classifier while giving consistent results
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
                os.path.join(DATA_PATH, df_dir, f"train_{i}.cands"),
                random_state=load_rng,
                shuffle=True,
            )
        ).drop(columns=drop_features if drop_features is not None else [])
        dev_df = create_index(
            load_candidates(
                os.path.join(DATA_PATH, df_dir, f"test_{i}.cands"),
                random_state=load_rng,
                shuffle=True,
            )
        ).drop(columns=drop_features if drop_features is not None else [])
        if train_first:
            clf = train_model(
                model,
                train_df,
                os.path.join(DATA_PATH, model_dir, f"{i}.joblib")
                if model_dir is not None
                else None,
            )
        else:
            clf = load(os.path.join(DATA_PATH, model_dir, f"{i}.joblib"))
        pred_tokens = predict_candidates(
            predict_probs(clf, dev_df),
            threshold=0.5,
        )
        comb_preds += predict_normalisation(
            test_raw,
            pred_tokens,
            os.path.join(DATA_PATH, output_dir, f"output_{i}.txt")
            if output_dir is not None
            else None,
            baseline_preds=mfr(train_raw, train_norm, test_raw) if with_mfr else None,
        )
    print("MFR")
    evaluate_predictions(comb_raw, comb_norm, mfr_preds)
    print("MODELS")
    return evaluate_predictions(comb_raw, comb_norm, comb_preds)[1:6]


def train_predict_evaluate(
    model,
    model_path,
    train_tweets_path,
    test_tweets_path,
    train_df_path,
    test_df_path,
    output_path=None,
    train_first=False,
    drop_features=None,
    with_mfr=False,
):
    load_rng = np.random.RandomState(42)
    train_df = load_candidates(train_df_path, random_state=load_rng, shuffle=True).drop(
        columns=drop_features if drop_features is not None else []
    )
    test_df = load_candidates(test_df_path, random_state=load_rng, shuffle=True).drop(
        columns=drop_features if drop_features is not None else []
    )
    raw, norm = loadNormData(test_tweets_path)
    if train_first:
        clf = train_model(model, train_df, model_path)
    else:
        clf = load(model_path)
    print("MFR")
    pred_tokens = predict_candidates(
        predict_probs(clf, test_df),
        threshold=0.5,
    )
    train_raw, train_norm = loadNormData(train_tweets_path)
    evaluate_predictions(raw, norm, mfr(train_raw, train_norm, raw))
    print("MODEL")
    predictions = predict_normalisation(
        raw,
        pred_tokens,
        output_path,
        baseline_preds=mfr(train_raw, train_norm, raw) if with_mfr else None,
    )
    return evaluate_predictions(raw, norm, predictions)[2]  # ERR


if __name__ == "__main__":
    params = {
        # "min_samples_leaf": 5,
        # "class_weight": "balanced",
        "max_depth": 10,
        "max_leaf_nodes": 100,
        "min_samples_split": 10,
    }
    model = create_rf(params, 100, random_state=np.random.RandomState(42))
    # # params = {"model__solver": "newton-cholesky"}
    # # model = create_logreg(params, np.random.RandomState(42))
    train_predict_evaluate(
        model,
        # os.path.join(DATA_PATH, "../models/rf.joblib"),
        None,
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "raw/test.norm"),
        os.path.join(DATA_PATH, "hpc/combined.cands"),
        os.path.join(DATA_PATH, "hpc/test.cands"),
        # os.path.join(DATA_PATH, "../models/output.txt"),
        None,
        train_first=True,
        # drop_features="cosine_to_orig",
        # with_mfr=True,
    )
    # train_predict_evaluate_cv(
    #     model,
    #     None,
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "hpc/cv"),
    #     None,
    #     # with_mfr=True
    #     # drop_features="orig_norms_seen",
    #     train_first=True,
    # )
    # feature_ablation(os.path.join(DATA_PATH, "hpc/feature_ablation.txt"))
