import math
import multiprocessing
import os
import pickle
from itertools import product
from multiprocessing import Process
from random import sample

import numpy as np
from sklearn import clone

from lexnorm.definitions import DATA_PATH
from lexnorm.models.classifiers import train_predict_evaluate_cv
from lexnorm.models.logreg import create_logreg
from lexnorm.models.random_forest import create_rf


def generate_configs(hyperparameters: dict):
    configs = list(product(*hyperparameters.values()))
    config_dicts = []
    for config in configs:
        config_dict = {}
        for i, param in enumerate(hyperparameters.keys()):
            config_dict[param] = config[i]
        config_dicts.append(config_dict)
    return config_dicts


def search(model, hyperparameters, tweets_path, df_dir):
    configs = generate_configs(hyperparameters)
    # configs = sample(configs, 100)
    results = {}
    for config in configs:
        new_model = clone(model)
        new_model.set_params(**config)
        err = train_predict_evaluate_cv(
            new_model, None, tweets_path, df_dir, None, True
        )
        print(config, err, flush=True)
        results[tuple(sorted(config.items()))] = err
    return results


if __name__ == "__main__":
    model = create_rf({}, 100, n_jobs=-1, random_state=np.random.RandomState(42))
    output = search(
        model,
        # {
        #     "max_depth": [5, 10, 15, None],
        #     "min_samples_leaf": [1, 10, 100, 1000],
        #     "min_samples_split": [2, 10, 100, 1000],
        #     "max_leaf_nodes": [10, 100, None],
        #     "class_weight": ["balanced", "balanced_subsample", None],
        #     "max_features": ["sqrt", "log2", None],
        # },
        {
            "max_depth": [7, 10, 13, 16],
            "min_samples_leaf": [1, 10, 20],
            "min_samples_split": [2, 6, 10, 14],
            "max_leaf_nodes": [100, None],
            "class_weight": [None],
            "max_features": [None],
        },
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "hpc/cv"),
    )
    # model = create_logreg({}, random_state=np.random.RandomState(42))
    # output = search(
    #     model,
    #     {
    #         "model__solver": ["newton-cholesky"],
    #         "model__penalty": ["l2"],
    #         "model__C": np.arange(0.3, 0, -0.01).tolist(),
    #         "model__class_weight": [None],
    #     },
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "hpc/cv"),
    # )
    with open(
        os.path.join(DATA_PATH, "processed/rf_hyperparams_refined.pickle"), "wb"
    ) as f:
        pickle.dump(output, f)
    # with open(os.path.join(DATA_PATH, "processed/hyperparams.pickle"), "rb") as f:
    #     output = pickle.load(f)
    # print(sorted(output.items(), key=lambda x: x[1], reverse=True))
