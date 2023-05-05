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
    queue.put((hyperparameters, err))


def hyperparameter_search(model, hyperparameters: dict, tweets_path, df_dir):
    configs = generate_configs(hyperparameters)
    print(configs)
    queue = multiprocessing.Queue()
    cores = multiprocessing.cpu_count()
    repetitions = math.ceil(len(configs) / cores)
    results = {}
    for i in range(repetitions):
        processes = []
        for j in range(cores):
            if len(configs) > i * cores + j:
                p = Process(
                    target=train_pred_eval_with_hyperparameters,
                    args=(model, configs[i * cores + j], tweets_path, df_dir, queue),
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


if __name__ == "__main__":
    model = create_rf({}, random_state=np.random.RandomState(42))
    output = (
        hyperparameter_search(
            model,
            {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5]},
            os.path.join(DATA_PATH, "processed/combined.txt"),
            os.path.join(DATA_PATH, "hpc/cv"),
        ),
    )
    with open(os.path.join(DATA_PATH, "processed/hyperparams.pickle"), "wb") as f:
        pickle.dump(output, f)
