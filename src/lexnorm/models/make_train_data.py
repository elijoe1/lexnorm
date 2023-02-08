import math
import multiprocessing
import os
import pickle
from multiprocessing import Process

import pandas as pd
from spylls.hunspell import Dictionary

from lexnorm.data import normEval, word2vec, norm_dict
from lexnorm.definitions import DATA_PATH
from lexnorm.models.candidate_generation import candidates_from_tweets

if __name__ == "__main__":
    # TODO refactor as a function which can be called for each cross validation fold (along with equivalent function with gold)
    """
    spawns 64 processes and gives each an equal number of tweets to process. output into multiprocessing queue, then
    when all processes finish, empty queue and write to csv
    """
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "interim/train.txt"))
    w2v = word2vec.get_vectors(os.path.join(DATA_PATH, "interim/train.txt"))
    with open(os.path.join(DATA_PATH, "interim/lexicon.txt"), "rb") as lf:
        lex = pickle.load(lf)
    normalisations = norm_dict.construct(os.path.join(DATA_PATH, "interim/train.txt"))
    spellcheck_dict = Dictionary.from_files("en_US")
    queue = multiprocessing.Queue()
    processes = []
    train_data = pd.DataFrame()
    batch_size = math.floor(len(raw) / 64)
    for i in range(0, 64):
        p = Process(
            target=candidates_from_tweets,
            args=(
                raw[i * batch_size : (i + 1) * batch_size],
                w2v,
                normalisations,
                lex,
                spellcheck_dict,
                queue,
            ),
        )
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        train_data = pd.concat([train_data, queue.get()])
    for p in processes:
        p.join()
    with open(os.path.join(DATA_PATH, "interim/candidates.txt"), "w") as f:
        train_data.to_csv(f)
