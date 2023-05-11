import math
import multiprocessing
import os
import pickle
from multiprocessing import Process

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from spylls.hunspell import Dictionary

from lexnorm.data import normEval, word2vec, norm_dict
from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.candidates import (
    candidates_from_tweets,
    add_ngram_orig_features,
    create_index,
)


def process_data_file(
    input_path: str, data_path: str, output_path: str = None, cores: int = 64
):
    raw_input, _ = normEval.loadNormData(input_path)
    raw_data, norm_data = normEval.loadNormData(data_path)
    return process_data(raw_input, raw_data, norm_data, output_path, cores)


def process_data(
    raw_input, raw_data, norm_data, output_path: str = None, cores: int = 64
):
    """
    spawns [cores] processes and gives each an equal number of tweets to process. output into multiprocessing queue, then
    when all processes finish, empty queue and write to csv
    """
    w2v = word2vec.get_vectors(raw_input + raw_data)
    normalisations = norm_dict.construct(raw_data, norm_data)
    with open(os.path.join(DATA_PATH, "processed/task_lexicon.pickle"), "rb") as lf:
        task_lex = pickle.load(lf)
    with open(os.path.join(DATA_PATH, "processed/feature_lexicon.pickle"), "rb") as lf:
        feature_lex = pickle.load(lf)
    spellcheck_dict = Dictionary.from_zip(
        os.path.join(DATA_PATH, "external/hunspell_en_US.zip")
    )
    queue = multiprocessing.Queue()
    processes = []
    output = pd.DataFrame()
    batch_size = math.ceil(len(raw_input) / cores)
    gold = True if raw_input == raw_data else False
    for i in range(cores):
        p = Process(
            target=candidates_from_tweets,
            args=(
                raw_input[i * batch_size : (i + 1) * batch_size],
                w2v,
                normalisations,
                task_lex,
                feature_lex,
                spellcheck_dict,
                queue,
                i,
                norm_data[i * batch_size : (i + 1) * batch_size] if gold else None,
            ),
        )
        processes.append(p)
    for p in processes:
        p.start()
    for _ in range(len(processes)):
        output = pd.concat([output, queue.get()])
    for p in processes:
        p.join()
    if output_path is not None:
        with open(output_path, "w") as f:
            output.to_csv(f)
    return output


def process_cv(data_path, output_dir):
    raw, norm = loadNormData(data_path)
    raw = np.array(raw, dtype=object)
    norm = np.array(norm, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, folds in enumerate(kf.split(raw, norm)):
        train, test = folds
        raw_train = raw[train].tolist()
        norm_train = norm[train].tolist()
        raw_test = raw[test].tolist()
        add_ngram_orig_features(
            process_data(raw_train, raw_train, norm_train),
            os.path.join(DATA_PATH, "processed"),
            os.path.join(output_dir, f"train_{i}.txt"),
        )
        add_ngram_orig_features(
            process_data(raw_test, raw_train, norm_train),
            os.path.join(DATA_PATH, "processed"),
            os.path.join(output_dir, f"test_{i}.txt"),
        )
        print(f"Completed {i+1}/5")


def process_train_test(train_path, test_path, train_output_path, test_output_path):
    create_index(
        add_ngram_orig_features(
            process_data_file(
                train_path,
                train_path,
            ),
            os.path.join(DATA_PATH, "processed"),
        ),
        output_path=train_output_path,
    )
    create_index(
        add_ngram_orig_features(
            process_data_file(
                test_path,
                train_path,
            ),
            os.path.join(DATA_PATH, "processed"),
        ),
        output_path=test_output_path,
    )


if __name__ == "__main__":
    # process_train_test(
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "raw/test.norm"),
    #     os.path.join(DATA_PATH, "hpc/combined.cands"),
    #     os.path.join(DATA_PATH, "hpc/test.cands"),
    # )
    process_cv(
        os.path.join(DATA_PATH, "processed/combined.txt"),
        os.path.join(DATA_PATH, "hpc/cv"),
    )
