import math
import multiprocessing
import os
import pickle
from itertools import product
from multiprocessing import Process

import numpy as np
from sklearn import clone

from lexnorm.definitions import DATA_PATH
from lexnorm.models.classifiers import train_predict_evaluate_cv
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


def train_pred_eval_with_hyperparameters(
    model, hyperparameters, tweets_path, df_dir, queue
):
    model = clone(model)
    model.set_params(**hyperparameters)
    err = train_predict_evaluate_cv(model, None, tweets_path, df_dir, None, True)
    print(hyperparameters, err)
    queue.put((hyperparameters, err))


def hyperparameter_search(model, hyperparameters: dict, tweets_path, df_dir):
    configs = generate_configs(hyperparameters)
    queue = multiprocessing.Queue()
    cores = multiprocessing.cpu_count() - 5  # leave some headroom
    repetitions = math.ceil(len(configs) / cores)
    results = {}
    for i in range(repetitions):
        processes = []
        for j in range(cores):
            if len(configs) > i * cores + j:
                config = configs[i * cores + j]
                print(config)
                p = Process(
                    target=train_pred_eval_with_hyperparameters,
                    args=(model, config, tweets_path, df_dir, queue),
                )
                processes.append(p)
        for p in processes:
            p.start()
        for _ in range(len(processes)):
            params, result = queue.get()
            results[tuple(sorted(params.items()))] = result
        for p in processes:
            p.join()
    return results


def search(model, hyperparameters, tweets_path, df_dir):
    configs = generate_configs(hyperparameters)
    results = {}
    for config in configs:
        new_model = clone(model)
        new_model.set_params(**config)
        err = train_predict_evaluate_cv(
            new_model, None, tweets_path, df_dir, None, True
        )
        print(config, err)
        results[tuple(sorted(config.items()))] = err
    return results


if __name__ == "__main__":
    model = create_rf({}, 100, n_jobs=-1, random_state=np.random.RandomState(42))
    output = search(
        model,
        {
            "max_depth": [5, 10, 15, None],
            "min_samples_leaf": [1, 10, 100, 1000],
            "min_samples_split": [2, 10, 100, 1000],
            "max_leaf_nodes": [10, 100, None],
            "class_weight": ["balanced", "balanced_subsample", None],
            "max_features": ["sqrt", "log2", None],
        },
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "hpc/cv"),
    )
    with open(os.path.join(DATA_PATH, "processed/hyperparams.pickle"), "wb") as f:
        pickle.dump(output, f)
    # with open(os.path.join(DATA_PATH, "processed/hyperparams.pickle"), "rb") as f:
    #     output = pickle.load(f)
    # print(sorted(output[0].items(), key=lambda x: x[1], reverse=True))
