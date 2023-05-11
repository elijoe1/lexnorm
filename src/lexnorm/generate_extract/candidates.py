import multiprocessing
import os
from collections import Counter
from typing import Optional

import gensim.models
import numpy as np
import spylls.hunspell
from spylls.hunspell import Dictionary

from lexnorm.data.word_ngrams import add_ngram_chunks
from lexnorm.definitions import DATA_PATH

from lexnorm.generate_extract.modules import *


def candidates_from_tweets(
    tweets: list[list[str]],
    vectors: gensim.models.KeyedVectors,
    normalisations: dict[str, Counter],
    task_lexicon: set,
    feature_lexicon: set,
    spellcheck_dictionary: spylls.hunspell.Dictionary,
    queue: multiprocessing.Queue,
    process: int,
    gold: Optional[list[list[str]]],
) -> None:
    """
    Target for producing a dataframe of candidates and extracted features. Candidates are produced for each raw token,
    and features are extracted. This be done in parallel as each tweet is independent. Training doesn't require any
    linking to the actual tokens, as each candidate/feature realisation is an independent example. However,
    'process', 'tok', and 'tweet' can do this for analysis. 'gold' and 'raw' are included for each candidate if possible
     to aid analysis.

    :param tweets: raw tweets
    :param vectors: word2vec word embeddings for word_embeddings module
    :param normalisations: normalisation dictionary for the norm_lookup module
    :param task_lexicon: lexicon for limiting generated candidates
    :param feature_lexicon: lexicon for orig_in_feature_lex feature
    :param spellcheck_dictionary: for the spellcheck module
    :param queue: to output dataframe into in a safe manner
    :param process: process number for token identification
    :param gold: normalised tweets for annotation, if annotated training data is desired
    """
    all_candidates = pd.DataFrame()
    tweet_index = 0
    for raw_tweet, norm_tweet in zip(tweets, gold if gold is not None else tweets):
        tok_index = 0
        for i, tok_pair in enumerate(zip(raw_tweet, norm_tweet)):
            raw_tok, norm_tok = tok_pair
            candidates = candidates_from_token(
                raw_tok,
                vectors,
                normalisations,
                task_lexicon,
                feature_lexicon,
                spellcheck_dictionary,
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
    task_lexicon: set,
    feature_lexicon: set,
    spellcheck_dictionary: Dictionary,
) -> pd.DataFrame:
    if not is_eligible(orig):
        return pd.DataFrame()
    candidates = original_token(orig)
    candidates = candidates.combine_first(spellcheck(orig, spellcheck_dictionary))
    candidates = candidates.combine_first(split(orig, task_lexicon))
    candidates = candidates.combine_first(clipping(orig, task_lexicon))
    # Obviously the norm_lookup module will always produce the correct candidate on the data it was constructed from
    candidates = candidates.combine_first(norm_lookup(orig, normalisations))
    candidates = candidates.combine_first(word_embeddings(orig, vectors, task_lexicon))

    # Any NaN values can be filled later
    def is_subseq(x, y):
        iterator = iter(y.lower())
        return all(c in iterator for c in x)

    # Generate/fill features
    candidates.cosine_to_orig = candidates.index.map(
        lambda x: vectors.similarity(x, orig)
        if (x in vectors and orig in vectors)
        else np.nan
    )
    candidates.spellcheck_score = candidates.index.map(
        lambda x: precise_affix_score(x, orig, -10, base=0, has_phonetic=False)
    )
    candidates["length"] = candidates.index.map(lambda x: len(x))
    candidates["frac_length"] = candidates["length"] / candidates.loc[orig]["length"]
    candidates["same_order"] = candidates.index.map(
        lambda x: 1 if is_subseq(orig, x) else np.nan
    )
    candidates["in_feature_lex_orig"] = 1 if orig in feature_lexicon else 0

    return candidates


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
