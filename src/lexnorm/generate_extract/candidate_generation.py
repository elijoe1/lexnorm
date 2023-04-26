import multiprocessing
from collections import Counter
from typing import Optional

import gensim.models
import numpy as np
import spylls.hunspell
from spylls.hunspell import Dictionary

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
    'process', 'tok', and 'tweet' can do this for analysis. 'gold' and 'raw' are included for each candidate
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
    # Copy features of original word to each candidate as decision whether to normalize based solely upon the original word.
    for feature in ["norms_seen", "frac_norms_seen", "length"]:
        candidates[f"orig_{feature}"] = candidates.loc[orig][feature]
    candidates["orig_in_feature_lex"] = 1 if orig in feature_lexicon else 0

    return candidates
