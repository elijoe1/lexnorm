import math
import multiprocessing
import os
import pickle
from collections import Counter
from multiprocessing import Process

import gensim.models
import numpy as np
import spylls.hunspell
from spylls.hunspell import Dictionary

from lexnorm.data import word2vec, norm_dict, normEval
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.modules import *


# def generate_candidates(
#     raw, vectors, lexicon, normalisations, spellcheck_dict
# ) -> list[list[list[str]]]:
#     """
#     Given an input tweet, produce a list of normalisation candidates for each token.
#
#     # TODO ability to exclude specific modules/steps?
#     # TODO implement generation for eligible tokens, not forgetting many-to-one normalisations
#     if we have no contextual features, we can generate candidates independently for each token - for each we can
#     extract features, then put all in a massive dataframe, labelling as correct or incorrect and train!!! So this
#     function has to, for a given tweet, produce all the candidates and their features per token. The candidate features
#     ust include the features of the original word as this is how we decide to normalise, and must be based on comparing
#     the candidate and the original token to evaluate how realistic normalising to is.
#
#     :param normalisations: normalisation dictionary for the normalisation lookup module
#     :param lexicon: lexicon for the split and clipping modules
#     :param vectors: vectors for the embeddings lookup module
#     :param tweet: to normalise, as list of tokens
#     :return: a corresponding list of sets of normalisation candidates
#     """

# TODO: ngram frequencies
def annotated_candidates_from_tweets(
    tweets: list[list[str]],
    vectors: gensim.models.KeyedVectors,
    normalisations: dict[str, Counter],
    lexicon: set,
    spellcheck_dictionary: spylls.hunspell.Dictionary,
    queue: multiprocessing.Queue,
    process: int,
    gold: list[list[str]],
) -> None:
    """
    Target for producing annotated training data, in the form of a dataframe of candidates and extracted features.
    Can be parallelised as each tweet is independent. Training doesn't require any linking to the actual tokens,
    as each candidate and extracted features is an independent example. However, 'process', 'tok', and 'tweet' can
    do this if necessary. 'gold' is included as well as 'correct' to make analysis easier.

    :param tweets: raw tweets
    :param vectors: word2vec word embeddings for word_embeddings module
    :param normalisations: normalisation dictionary for the norm_lookup module
    :param lexicon: lexicon for the split and clipping modules
    :param spellcheck_dictionary: for the spellcheck module
    :param queue: to output dataframe into in a safe manner
    :param process: process number for token identification
    :param gold: normalised tweets for annotation
    """
    all_candidates = pd.DataFrame()
    tweet_index = 0
    tok_index = 0
    for raw_tweet, norm_tweet in zip(tweets, gold):
        for raw_tok, norm_tok in zip(raw_tweet, norm_tweet):
            candidates = candidates_from_token(
                raw_tok, vectors, normalisations, lexicon, spellcheck_dictionary
            )
            if not candidates.empty:
                candidates["correct"] = candidates.index.map(
                    lambda x: 1 if x == norm_tok else np.nan
                )
                candidates["process"] = process
                candidates["tweet"] = tweet_index
                candidates["tok"] = tok_index
                candidates["gold"] = norm_tok
            tok_index += 1
            all_candidates = pd.concat([all_candidates, candidates])
        tweet_index += 1
    queue.put(all_candidates)


def candidates_from_tweets(
    tweets, vectors, normalisations, lexicon, spellcheck_dictionary, queue, process
):
    all_candidates = pd.DataFrame()
    tweet_index = 0
    tok_index = 0
    for tweet in tweets:
        for tok in tweet:
            candidates = candidates_from_token(
                tok, vectors, normalisations, lexicon, spellcheck_dictionary
            )
            if not candidates.empty:
                candidates["process"] = process
                candidates["tweet"] = tweet_index
                candidates["tok"] = tok_index
            tok_index += 1
            all_candidates = pd.concat([all_candidates, candidates])
        tweet_index += 1
    queue.put(all_candidates)


def candidates_from_token(
    orig: str,
    vectors: gensim.models.KeyedVectors,
    normalisations: dict[str, Counter],
    lexicon: set,
    spellcheck_dictionary: Dictionary,
) -> pd.DataFrame:
    if not is_eligible(orig):
        return pd.DataFrame()
    candidates = original_token(orig)
    candidates = candidates.combine_first(word_embeddings(orig, vectors))
    candidates = candidates.combine_first(norm_lookup(orig, normalisations))
    candidates = candidates.combine_first(spellcheck(orig, spellcheck_dictionary))
    # obviously lookup on the train set will always produce the correct candidate (perhaps among others)!
    candidates = candidates.combine_first(clipping(orig, lexicon))
    candidates = candidates.combine_first(split(orig, lexicon))
    # can set incalculable similarities to NaN as this is automatically picked up by the rand forest - MAY WANT TO BE
    # DIFFERENT for other models.
    candidates.cosine_to_orig = candidates.index.map(
        lambda x: vectors.similarity(x, orig)
        if (x in vectors and orig in vectors)
        else np.nan
    )
    candidates["in_lexicon"] = candidates.index.map(
        lambda x: 1 if x in lexicon else np.nan
    )
    candidates["length"] = candidates.index.map(lambda x: len(x))
    candidates["same_order"] = candidates.index.map(
        lambda x: 1 if is_subseq(orig, x) else np.nan
    )
    # copy across features of original word to each candidate as decision whether to normalize based solely upon the original word
    # do we really want to copy same_order? done in MoNoise but suspicious...literally has no effect
    for feature in ["norms_seen", "in_lexicon", "same_order", "length"]:
        candidates[f"orig_{feature}"] = candidates.loc[orig][feature]
    # TODO internally calculated distance and rank for hunspell (if even possible)
    # TODO ngram probabilities
    # TODO freq of cand in train?
    return candidates


def is_subseq(x, y):
    iterator = iter(y)
    return all(c in iterator for c in x)


if __name__ == "__main__":
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "raw/train.norm"))
    w2v = word2vec.get_vectors(os.path.join(DATA_PATH, "raw/train.norm"))
    with open(os.path.join(DATA_PATH, "interim/lexicon.txt"), "rb") as lf:
        lex = pickle.load(lf)
    normalisations = norm_dict.construct(os.path.join(DATA_PATH, "raw/train.norm"))
    spellcheck_dict = Dictionary.from_files("en_US")
    queue = multiprocessing.Queue()
    processes = []
    train_data = pd.DataFrame()
    batch_size = math.ceil(len(raw) / 64)
    for i in range(0, 64):
        p = Process(
            target=annotated_candidates_from_tweets,
            # target=candidates_from_tweets,
            args=(
                raw[i * batch_size : (i + 1) * batch_size],
                w2v,
                normalisations,
                lex,
                spellcheck_dict,
                queue,
                i,
                norm[i * batch_size : (i + 1) * batch_size],
            ),
        )
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        train_data = pd.concat([train_data, queue.get()])
    for p in processes:
        p.join()
    with open(os.path.join(DATA_PATH, "hpc/train_annotated.txt"), "w+") as f:
        train_data.to_csv(f)
