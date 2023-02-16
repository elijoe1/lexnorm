import math
import multiprocessing
import os
import pickle
from multiprocessing import Process

import pandas as pd
from spylls.hunspell import Dictionary

from lexnorm.data import normEval, word2vec, norm_dict
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.candidate_generation import candidates_from_tweets


def process_data(input_path: str, data_path: str, output_path: str, cores: int = 64):
    """
    spawns cores processes and gives each an equal number of tweets to process. output into multiprocessing queue, then
    when all processes finish, empty queue and write to csv
    """
    raw, norm = normEval.loadNormData(input_path)
    # TODO: actually could construct vectors from both input and data path as unsupervised
    w2v = word2vec.get_vectors(data_path)
    normalisations = norm_dict.construct(data_path)
    with open(os.path.join(DATA_PATH, "interim/lexicon.txt"), "rb") as lf:
        lex = pickle.load(lf)
    spellcheck_dict = Dictionary.from_files("en_US")
    queue = multiprocessing.Queue()
    processes = []
    output = pd.DataFrame()
    batch_size = math.ceil(len(raw) / cores)
    gold = True if input_path == data_path else False
    for i in range(cores):
        p = Process(
            target=candidates_from_tweets,
            args=(
                raw[i * batch_size : (i + 1) * batch_size],
                w2v,
                normalisations,
                lex,
                spellcheck_dict,
                queue,
                i,
                norm[i * batch_size : (i + 1) * batch_size] if gold else None,
            ),
        )
        processes.append(p)
    for p in processes:
        p.start()
    for _ in range(len(processes)):
        output = pd.concat([output, queue.get()])
    for p in processes:
        p.join()
    with open(output_path, "w+") as f:
        output.to_csv(f)


if __name__ == "__main__":
    process_data(
        os.path.join(DATA_PATH, "raw/train.norm"),
        os.path.join(DATA_PATH, "raw/train.norm"),
        os.path.join(DATA_PATH, "hpc/train_processed_annotated.txt"),
    )
    process_data(
        os.path.join(DATA_PATH, "raw/dev.norm"),
        os.path.join(DATA_PATH, "raw/train.norm"),
        os.path.join(DATA_PATH, "hpc/dev_processed.txt"),
    )
