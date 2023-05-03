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
from lexnorm.data.word_ngrams import add_ngram_chunks
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.candidate_generation import candidates_from_tweets
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.normalise import load_candidates


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


def add_ngram_orig_features(
    dataframe, ngram_counter_path=os.path.join(DATA_PATH, "processed"), output_path=None
):
    """
    Adds ngram features to a dataframe, namely unigram probabilities of candidate and bigram probabilities candidate given
    previous and next word.

    :param output_path: Output path for updated dataframe, if desired
    :param dataframe: Dataframe to add ngram features to
    :param ngram_counter_path: Path to csvs of ngram dataframes
    """
    dataframe = dataframe.copy().reset_index(names="cand")
    for domain in ["wiki", "twitter"]:
        for tok_pos in ["cand", "prev", "next"]:
            dataframe = dataframe.set_index(dataframe[tok_pos], drop=False)
            dataframe = add_ngram_chunks(
                dataframe,
                os.path.join(ngram_counter_path, f"{domain}_unigrams.ngr"),
                f"{domain}_uni_{tok_pos}",
                True,
            )
    for domain in ["wiki", "twitter"]:
        for tok_poss in [("prev", "cand"), ("cand", "next")]:
            dataframe = dataframe.set_index(
                dataframe[tok_poss[0]] + " " + dataframe[tok_poss[1]], drop=False
            )
            dataframe = add_ngram_chunks(
                dataframe,
                os.path.join(ngram_counter_path, f"{domain}_bigrams.ngr"),
                f"{domain}_bi_{tok_poss[0]}_{tok_poss[1]}",
                True,
            )
    for domain in ["wiki", "twitter"]:
        total = dataframe[f"{domain}_uni_cand"].sum()
        dataframe[f"{domain}_bi_prev_cand"] /= dataframe[f"{domain}_uni_prev"]
        dataframe[f"{domain}_bi_cand_next"] /= dataframe[f"{domain}_uni_next"]
        dataframe[f"{domain}_uni_cand"] /= total
    dataframe = dataframe.drop(
        columns=[
            "wiki_uni_prev",
            "wiki_uni_next",
            "twitter_uni_prev",
            "twitter_uni_next",
        ]
    )
    dataframe = dataframe.merge(
        dataframe.loc[dataframe.from_original_token == 1][
            [
                "twitter_uni_cand",
                "twitter_bi_prev_cand",
                "twitter_bi_cand_next",
                "wiki_uni_cand",
                "wiki_bi_prev_cand",
                "wiki_bi_cand_next",
                "length",
                "norms_seen",
                "frac_norms_seen",
                "process",
                "tweet",
                "tok",
            ]
        ],
        "left",
        on=["process", "tweet", "tok"],
        suffixes=(None, "_orig"),
    )
    dataframe = dataframe.set_index("cand")
    if output_path is not None:
        with open(output_path, "w") as f:
            dataframe.to_csv(f)
    return dataframe


def create_index(dataframe, offset=0, output_path=None):
    """
    Replaces "process", "tweet", "tok" columns with "tok_id" column which gives an index into the list of eligible tokens
    in the dataset used to produce the dataframe of the corresponding raw token. (NOTE: not index into list of all tokens)

    :param dataframe: A dataframe of candidates and extracted features.
    :param output_path: A path to save the new dataframe, if desired.
    :param offset: Offset for index, if creating index over multiple dataframes
    :return: The new dataframe.
    """
    data = dataframe.copy()
    data = data.sort_values(["process", "tweet", "tok"])
    data["tok_id"] = (
        data.sort_values(["process", "tweet", "tok"])
        .groupby(["process", "tweet", "tok"])
        .ngroup()
    ) + offset
    data = data.drop(["process", "tweet", "tok"], axis=1)
    if output_path is not None:
        with open(output_path, "w") as f:
            data.to_csv(f)
    return data


def link_to_gold(dataframe, raw_gold, norm_gold, output_path=None):
    """
    Adds "gold" and "correct" columns to candidates dataframe with "tok_id" column by getting list of normalisations
    of eligible tokens using raw_gold and norm_gold, and using tok_id as an index into the list. This is useful for analysis.

    :param norm_gold: gold raw tweets
    :param raw_gold: gold norm tweets
    :param dataframe: Dataframe to add gold and correct column to
    :param output_path: Path to save new dataframe to, if desired
    :return:Dataframe with columns added
    """
    dataframe = dataframe.copy()
    eligible_norms = []
    for raw_tweet, norm_tweet in zip(raw_gold, norm_gold):
        for raw_tok, norm_tok in zip(raw_tweet, norm_tweet):
            if is_eligible(raw_tok):
                eligible_norms.append(norm_tok)
    dataframe["gold"] = dataframe.apply(lambda x: eligible_norms[x.tok_id], axis=1)
    dataframe["correct"] = dataframe.index.values == dataframe.gold
    if output_path is not None:
        with open(output_path, "w") as f:
            dataframe.to_csv(f)
    return dataframe


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
    # process_data_file(
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "hpc/fixed_train.norm"),
    # )
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
    # c = load_candidates(os.path.join(DATA_PATH, "hpc/train_final.cands"))
    # c = c.set_index("cand.1")
    # c.to_csv(os.path.join(DATA_PATH, "hpc/train_final.cands"))
    # create_index(
    #     c, output_path=os.path.join(DATA_PATH, "hpc/fixed_dev_ngrams_idx.norm")
    # )
    # c = load_candidates(os.path.join(DATA_PATH, "hpc/fixed_train_ngrams.norm"))
    # create_index(
    #     c,
    #     output_path=os.path.join(DATA_PATH, "hpc/fixed_train_ngrams_idx.norm"),
    # )
