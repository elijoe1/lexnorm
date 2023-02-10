import os

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from lexnorm.definitions import DATA_PATH


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(DATA_PATH, "hpc/train_annotated.txt"), index_col=0)
    train_X = train.fillna(0).drop(
        ["correct", "gold", "process", "tweet", "tok"], axis=1
    )
    train_y = train.fillna(0)["correct"]
    rf_clf = RandomForestClassifier(
        n_jobs=-1, random_state=42, verbose=1, class_weight="balanced"
    )
    rf_clf.fit(train_X, train_y)
    dump(rf_clf, os.path.join(DATA_PATH, "../models/rf.joblib"))
