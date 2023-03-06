import math
import multiprocessing
import os
import pickle
from collections import Counter
from multiprocessing import Process

import pandas as pd
from spylls.hunspell import Dictionary

from lexnorm.data import normEval, word2vec, norm_dict
from lexnorm.data.word_ngrams import counter_from_pickle
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.candidate_generation import candidates_from_tweets
from lexnorm.generate_extract.filtering import is_eligible


def process_data(input_path: str, data_path: str, output_path: str, cores: int = 64):
    """
    spawns [cores] processes and gives each an equal number of tweets to process. output into multiprocessing queue, then
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


def add_ngram_features(dataframe_path, ngram_counter_path, output_path=None):
    """
    Adds ngram features to a dataframe, namely unigram probabilities of candidate and bigram probabilities of previous
    and next word given candidate.

    Issue: unseen unigrams in the test set. This is presumed quite unlikely as the models are so large, but still
    requires thought. Hacky solution done by VDG is to give unseen unigrams a frequency of 1 - bigrams freqs will
    obviously still be 0 so this only affects unigram probs (frequencies).
    TODO: investigate the frequency of this in the data.
    TODO: may have to parallelise this.

    Not doing bigram smoothing as run into size of vocabulary issue if doing laplacian smoothing - no predefined size.

    :param output_path: Output path for updated dataframe, if desired
    :param dataframe_path: Path of dataframe to add ngram features to
    :param ngram_counter_path: Path to pickles of ngram counters
    """
    dataframe = pd.read_csv(
        dataframe_path,
        index_col=0,
        keep_default_na=False,
        na_values="",
    )
    twitter_unigram_counter = counter_from_pickle(
        os.path.join(ngram_counter_path, "twitter_unigram_counter.pickle")
    )
    twitter_bigram_counter = counter_from_pickle(
        os.path.join(ngram_counter_path, "twitter_bigram_counter.pickle")
    )
    wiki_unigram_counter = counter_from_pickle(
        os.path.join(ngram_counter_path, "wiki_unigram_counter.pickle")
    )
    wiki_bigram_counter = counter_from_pickle(
        os.path.join(ngram_counter_path, "wiki_bigram_counter.pickle")
    )
    dataframe["twitter_uni"] = dataframe.index.map(
        lambda x: twitter_unigram_counter.get(x, 1)
    )
    # next_uni = dataframe.apply(lambda x: twitter_unigram_counter.get(x.next, 1), axis=1)
    dataframe["twitter_bi_prev"] = dataframe.apply(
        lambda x: twitter_bigram_counter.get(" ".join([x.prev, x.name]), 0)
        / x.twitter_uni,
        axis=1,
    )
    dataframe["twitter_bi_next"] = dataframe.apply(
        lambda x: twitter_bigram_counter.get(" ".join([x.name, x.next]), 0)
        / x.twitter_uni,
        axis=1,
    )
    # dataframe["twitter_bi_next"] /= next_uni
    dataframe["wiki_uni"] = dataframe.index.map(
        lambda x: wiki_unigram_counter.get(x, 1)
    )
    # next_uni = dataframe.apply(lambda x: wiki_unigram_counter.get(x.next, 1), axis=1)
    dataframe["wiki_bi_prev"] = dataframe.apply(
        lambda x: wiki_bigram_counter.get(" ".join([x.prev, x.name]), 0) / x.wiki_uni,
        axis=1,
    )
    dataframe["wiki_bi_next"] = dataframe.apply(
        lambda x: wiki_bigram_counter.get(" ".join([x.name, x.next]), 0) / x.wiki_uni,
        axis=1,
    )
    # dataframe["wiki_bi_next"] /= next_uni
    if output_path is not None:
        with open(output_path, "w+") as f:
            dataframe.to_csv(f)
    return dataframe


def create_index(dataframe, output_path=None):
    """
    Replaces "process", "tweet", "tok" columns with "tok_id" column which gives an index into the list of eligible tokens
    in the dataset used to produce the dataframe of the corresponding raw token. (NOTE: not index into list of all tokens)

    :param dataframe: A dataframe of candidates and extracted features.
    :param output_path: A path to save the new dataframe, if desired.
    :return: The new dataframe.
    """
    data = dataframe.copy()
    data = data.sort_values(["process", "tweet", "tok"])
    data["tok_id"] = (
        data.sort_values(["process", "tweet", "tok"])
        .groupby(["process", "tweet", "tok"])
        .ngroup()
    )
    data = data.drop(["process", "tweet", "tok"], axis=1)
    if output_path is not None:
        with open(output_path, "w+") as f:
            data.to_csv(f)
    return data


def link_to_gold(dataframe, gold_path):
    """
    Adds "gold" and "correct" column to candidates dataframe with "tok_id" column by getting list of normalisations
    of eligible tokens using gold_path, and using tok_id as an index into the list. This is useful for analysis.

    :param dataframe: Dataframe to add gold and correct column to
    :param gold_path: Path to data to link gold normalisations from
    :return: Tuple of dataframe with columns added
    """
    dataframe = dataframe.copy()
    raw, norm = normEval.loadNormData(gold_path)
    eligible_norms = []
    for raw_tweet, norm_tweet in zip(raw, norm):
        for raw_tok, norm_tok in zip(raw_tweet, norm_tweet):
            if is_eligible(raw_tok):
                eligible_norms.append(norm_tok)
    dataframe["gold"] = dataframe.apply(lambda x: eligible_norms[x.tok_id], axis=1)
    dataframe["correct"] = dataframe.index.values == dataframe.gold
    return dataframe


if __name__ == "__main__":
    # process_data(
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "hpc/train_processed_annotated_nocap_neighbours.txt"),
    # )
    # process_data(
    #     os.path.join(DATA_PATH, "raw/dev.norm"),
    #     os.path.join(DATA_PATH, "raw/train.norm"),
    #     os.path.join(DATA_PATH, "hpc/dev_processed_nocap_neighbours.txt"),
    # )
    add_ngram_features(
        os.path.join(DATA_PATH, "hpc/train_processed_annotated_nocap_neighbours.txt"),
        os.path.join(DATA_PATH, "processed"),
        os.path.join(DATA_PATH, "hpc/train_ngrams.txt"),
    )
    add_ngram_features(
        os.path.join(DATA_PATH, "hpc/dev_processed_nocap_neighbours.txt"),
        os.path.join(DATA_PATH, "processed"),
        os.path.join(DATA_PATH, "hpc/dev_ngrams.txt"),
    )
