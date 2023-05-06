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


def create_grid(hyperparameters: dict):
    configs = list(product(*hyperparameters.values()))
    config_dicts = []
    for config in configs:
        config_dict = {}
        for i, param in enumerate(hyperparameters.keys()):
            config_dict[param] = config[i]
        config_dicts.append(config_dict)
    return config_dicts


def create_random(hyperparameters: dict, n_samples=100):
    config_dicts = []
    for _ in range(n_samples):
        config_dict = {}
        for k, v in hyperparameters.items():
            config_dict[k] = sample(v, 1)[0]
        config_dicts.append(config_dict)
    return config_dicts


def search(model, hyperparameters, tweets_path, df_dir, type="grid"):
    if type == "grid":
        configs = create_grid(hyperparameters)
    else:
        configs = create_random(hyperparameters)
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
        # {
        #     "max_depth": [7, 10, 13, 16],
        #     "min_samples_leaf": [1, 10, 20],
        #     "min_samples_split": [2, 6, 10, 14],
        #     "max_leaf_nodes": [100, None],
        #     "class_weight": [None],
        #     "max_features": [None],
        # },
        {
            "max_depth": list(range(1, 20)),
            "min_samples_leaf": list(range(1, 20)),
            "min_samples_split": list(range(2, 20)),
            "max_leaf_nodes": list(range(50, 300)),
        },
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "hpc/cv"),
        type="random",
    )
    # model = create_logreg({}, random_state=np.random.RandomState(42))
    # output = search(
    #     model,
    #     {
    #         "model__solver": ["newton-cholesky"],
    #         "model__penalty": ["l2"],
    #         "model__C": np.arange(0.03, 0, -0.001).tolist(),
    #         "model__class_weight": [None],
    #     },
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "hpc/cv"),
    # )
    with open(
        os.path.join(DATA_PATH, "processed/rf_hyperparams_refined_smallrange.pickle"),
        "wb",
    ) as f:
        pickle.dump(output, f)
    # with open(
    #     os.path.join(DATA_PATH, "processed/rf_hyperparams_refined.pickle"),
    #     "rb",
    # ) as f:
    #     output = pickle.load(f)
    # print(sorted(output.items(), key=lambda x: x[1][1], reverse=True))
