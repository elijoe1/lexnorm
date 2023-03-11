import os

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold

from lexnorm.data.baseline import mfr
from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.evaluation.predictions import evaluate_predictions
from lexnorm.generate_extract.process import create_index

from lexnorm.models.normalise import prep_train, normalise, load_candidates
from lexnorm.models.predict import predict_probs, predict_normalisations


def train_rf(candidates, random_state=None, output_path=None):
    train_X, train_y = prep_train(candidates)
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        # class_weight="balanced",
        oob_score=True,
        # suggested by scikit docs as good to prevent over-fitting (seems to work a little...?)
        # pre-pruning
        min_samples_leaf=5
        # prevent over-fitting
        # turns out don't need this with abstaining if low prob. Brings precision down, recall up.
        # HIGHER THAN MFR!!
        # max_depth=3,
    )
    rf_clf.fit(train_X, train_y)
    if output_path is not None:
        dump(rf_clf, output_path)
    return rf_clf


def train_adaboost(candidates, random_state=None, output_path=None):
    train_X, train_y = prep_train(candidates)
    rf_clf = AdaBoostClassifier(n_estimators=200, random_state=random_state)
    rf_clf.fit(train_X, train_y)
    if output_path is not None:
        dump(rf_clf, output_path)
    return rf_clf


def train_predict_evaluate_cv(
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
        ).drop(columns=drop_features if drop_features is not None else [])
        dev_df = create_index(
            load_candidates(
                os.path.join(DATA_PATH, df_dir, f"test_{i}.txt"),
                random_state=load_rng,
                shuffle=True,
            )
        ).drop(columns=drop_features if drop_features is not None else [])
        if train_first:
            clf = train_rf(
                train_df,
                model_rng,
                os.path.join(DATA_PATH, model_dir, f"rf_{i}.joblib")
                if model_dir is not None
                else None,
            )
        else:
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
            baseline_preds=mfr(train_raw, train_norm, test_raw) if with_mfr else None,
        )
    evaluate_predictions(comb_raw, comb_norm, mfr_preds)
    return evaluate_predictions(comb_raw, comb_norm, comb_preds)[2]  # ERR


def train_predict_evaluate(
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
    # NOTE assumes create_index already saved to file, as is fine with non-cv operation
    model_rng = np.random.RandomState(42)
    load_rng = np.random.RandomState(42)
    train_df = load_candidates(train_df_path, random_state=load_rng, shuffle=True).drop(
        columns=drop_features if drop_features is not None else []
    )
    test_df = load_candidates(test_df_path, random_state=load_rng, shuffle=True).drop(
        columns=drop_features if drop_features is not None else []
    )
    raw, norm = loadNormData(test_tweets_path)
    if train_first:
        clf = train_rf(train_df, model_rng, model_path)
    else:
        clf = load(model_path)
    pred_tokens = predict_normalisations(
        predict_probs(clf, test_df),
        threshold=0.5,
    )
    train_raw, train_norm = loadNormData(train_tweets_path)
    evaluate_predictions(raw, norm, mfr(train_raw, train_norm, raw))
    predictions = normalise(
        raw,
        pred_tokens,
        output_path,
        baseline_preds=mfr(train_raw, train_norm, raw) if with_mfr else None,
    )
    return evaluate_predictions(raw, norm, predictions)[2]  # ERR


def feature_ablation(output_path):
    features = [
        "cosine_to_orig",
        "embeddings_rank",
        "from_clipping",
        "from_original_token",
        "from_split",
        "norms_seen",
        "spellcheck_rank",
        "in_lexicon",
        "length",
        "same_order",
        "orig_norms_seen",
        "orig_in_lexicon",
        # TODO remove
        "orig_same_order",
        "orig_length",
        "twitter_uni",
        "twitter_bi_prev",
        "twitter_bi_next",
        "wiki_uni",
        "wiki_bi_prev",
        "wiki_bi_next",
    ]
    scores = {}
    for feature in features:
        # TODO make this non cv version and do on combined and test
        scores[feature] = train_predict_evaluate_cv(
            None,
            os.path.join(DATA_PATH, "processed/combined.txt"),
            os.path.join(DATA_PATH, "hpc/cv"),
            None,
            train_first=True,
            drop_features=feature,
        )
    with open(output_path, "w") as f:
        f.write(str(scores))


if __name__ == "__main__":
    # train_predict_evaluate(
    #     os.path.join(DATA_PATH, "../models/rf.joblib"),
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "raw/dev.norm"),
    #     os.path.join(DATA_PATH, "hpc/train_pipeline.txt"),
    #     os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"),
    #     os.path.join(DATA_PATH, "../models/output.txt"),
    #     train_first=True,
    #     # drop_features="cosine_to_orig",
    #     # with_mfr=True,
    # )
    train_predict_evaluate_cv(
        os.path.join(DATA_PATH, "../models"),
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "hpc/cv"),
        os.path.join(DATA_PATH, "../models/output"),
        # with_mfr=True
        # drop_features="orig_norms_seen",
        train_first=True,
    )
    # feature_ablation(os.path.join(DATA_PATH, "hpc/feature_ablation.txt"))
