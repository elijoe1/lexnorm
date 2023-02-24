import math
import multiprocessing
import os
import pickle
from collections import Counter
from multiprocessing import Process
from typing import Optional

import gensim.models
import numpy as np
import spylls.hunspell
from spylls.hunspell import Dictionary

from lexnorm.data import word2vec, norm_dict, normEval
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.modules import *


def candidates_from_tweets(
    tweets: list[list[str]],
    vectors: gensim.models.KeyedVectors,
    normalisations: dict[str, Counter],
    lexicon: set,
    spellcheck_dictionary: spylls.hunspell.Dictionary,
    queue: multiprocessing.Queue,
    process: int,
    gold: Optional[list[list[str]]],
) -> None:
    """
    Target for producing a dataframe of candidates and extracted features. Candidates are produced for each raw token,
    and features are extracted. This be done in parallel as each tweet is independent. Training doesn't require any
    linking to the actual tokens, as each candidate/feature realisation is an independent example. However,
    'process', 'tok', and 'tweet' can do this for analysis. 'gold' and 'raw' are included for each candidate
     to aid analysis.

    :param tweets: raw tweets
    :param vectors: word2vec word embeddings for word_embeddings module
    :param normalisations: normalisation dictionary for the norm_lookup module
    :param lexicon: lexicon for the split and clipping modules
    :param spellcheck_dictionary: for the spellcheck module
    :param queue: to output dataframe into in a safe manner
    :param process: process number for token identification
    :param gold: normalised tweets for annotation, if annotated training data is desired
    """
    # TODO ability to exclude specific generation modules?
    # TODO bigram probabilities
    # TODO many to one normalisations?
    all_candidates = pd.DataFrame()
    tweet_index = 0
    for raw_tweet, norm_tweet in zip(tweets, gold if gold is not None else tweets):
        tok_index = 0
        for i, tok_pair in enumerate(zip(raw_tweet, norm_tweet)):
            raw_tok, norm_tok = tok_pair
            candidates = candidates_from_token(
                raw_tok, vectors, normalisations, lexicon, spellcheck_dictionary
            )
            if not candidates.empty:
                candidates["raw"] = raw_tok
                candidates["prev"] = "<s>" if i == 0 else raw_tweet[i - 1]
                candidates["next"] = (
                    "</s>" if i == len(raw_tweet) - 1 else raw_tweet[i + 1]
                )
                if gold is not None:
                    candidates["gold"] = norm_tok
                    candidates["correct"] = candidates.index.map(
                        lambda x: 1 if x == norm_tok else np.nan
                    )
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
    candidates = candidates.combine_first(spellcheck(orig, spellcheck_dictionary))
    candidates = candidates.combine_first(split(orig, lexicon))
    candidates = candidates.combine_first(clipping(orig, lexicon))
    # Obviously the norm_lookup module will always produce the correct candidate on the data it was constructed from
    candidates = candidates.combine_first(norm_lookup(orig, normalisations))
    candidates = candidates.combine_first(word_embeddings(orig, vectors))

    # Can set missing values to NaN and fill later as this is automatically picked up by the random forest
    # This wil NOT be the case for other models e.g. logistic regression.
    def is_subseq(x, y):
        iterator = iter(y.lower())
        return all(c in iterator for c in x)

    # Generate/fill features
    candidates.cosine_to_orig = candidates.index.map(
        lambda x: vectors.similarity(x, orig)
        if (x in vectors and orig in vectors)
        else np.nan
    )
    candidates["in_lexicon"] = candidates.index.map(
        lambda x: 1 if x.lower() in lexicon else np.nan
    )
    candidates["length"] = candidates.index.map(lambda x: len(x))
    candidates["same_order"] = candidates.index.map(
        lambda x: 1 if is_subseq(orig, x) else np.nan
    )
    # Copy features of original word to each candidate as decision whether to normalize based solely upon the original word.
    # TODO: do we really want to copy same_order? Done in MoNoise but suspicious...is constant
    for feature in ["norms_seen", "in_lexicon", "same_order", "length"]:
        candidates[f"orig_{feature}"] = candidates.loc[orig][feature]
    # TODO internally calculated distance for spellcheck (not possible in this implementation of hunspell)
    # TODO unigram probabilities
    # TODO frequency of candidate in train - that is sum of all normalisations seen?
    # TODO percentage of normalisations seen?
    return candidates
