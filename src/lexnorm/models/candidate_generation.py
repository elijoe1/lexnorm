import math
import multiprocessing
import os
import pickle
from collections import Counter
from multiprocessing import Process

import gensim.models
import numpy as np
import pandas as pd
from spylls.hunspell import Dictionary

from lexnorm.data import word2vec, norm_dict, normEval
from lexnorm.definitions import DATA_PATH
from lexnorm.models.filtering import is_eligible


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
#     candidates_list = []
#     for raw_tweet in raw:
#         tweet_candidates = []
#         for i, tok in enumerate(raw_tweet):
#             if tok == "rt":
#                 # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
#                 # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
#                 # of tweet and not followed by @mention) and when normalised, always to 'retweet'
#                 if 0 < i < len(raw_tweet) - 1 and raw_tweet[i + 1][0] != "@":
#                     tweet_candidates.append(["retweet"])
#                 else:
#                     tweet_candidates.append([tok])
#             elif is_eligible(tok):
#                 candidates = set()
#                 candidates = candidates.union(original_token(tok))
#                 candidates = candidates.union(word_embeddings(tok, vectors))
#                 candidates = candidates.union(spellcheck(tok, spellcheck_dict))
#                 # obviously lookup on the train set will always produce the correct candidate (perhaps among others)!
#                 candidates = candidates.union(lookup(tok, normalisations))
#                 candidates = candidates.union(clipping(tok, lexicon))
#                 candidates = candidates.union(split(tok, lexicon))
#                 tweet_candidates.append(list(candidates))
#             else:
#                 tweet_candidates.append([tok])
#         candidates_list.append(tweet_candidates)
#     return candidates_list


def annotated_candidates_from_tweets(
    tweets,
    vectors,
    normalisations,
    lexicon,
    spellcheck_dictionary,
    queue,
    process,
    gold,
):
    all_candidates = pd.DataFrame()
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
                candidates["raw_tok_index"] = f"{process}_{tok_index}"
                candidates["gold"] = norm_tok
                tok_index += 1
            all_candidates = pd.concat([all_candidates, candidates])
    queue.put(all_candidates)


def candidates_from_tweets(
    tweets, vectors, normalisations, lexicon, spellcheck_dictionary, queue
):
    all_candidates = pd.DataFrame()
    for tweet in tweets:
        for tok in tweet:
            all_candidates = pd.concat(
                [
                    all_candidates,
                    candidates_from_token(
                        tok, vectors, normalisations, lexicon, spellcheck_dictionary
                    ),
                ]
            )
    queue.put(all_candidates)


def candidates_from_token(
    orig: str,
    vectors: gensim.models.keyedvectors,
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
    # do we really want to copy same_order? done in MoNoise but suspicious...
    for feature in ["norms_seen", "in_lexicon", "same_order", "length"]:
        candidates[f"orig_{feature}"] = candidates.loc[orig][feature]
    # TODO internally calculated distance and rank for hunspell (if even possible)
    # TODO ngram probabilities
    # TODO freq of cand in train?
    return candidates


def is_subseq(x, y):
    iterator = iter(y)
    return all(c in iterator for c in x)


def original_token(tok):
    # AS IN MONOISE
    # needed if detect step is skipped, as all tokens will be replaced by one from the list of candidates
    candidate = pd.DataFrame(columns=["from_original_token"])
    candidate.loc[tok] = {"from_original_token": 1}
    return candidate


def word_embeddings(tok, vectors, threshold=0):
    # TODO uni, bigram freqs?
    # TODO implement word2vec with keras. Experiment with different no. of candidates generated.
    #  Could even create twitter embeddings myself? Could clean up as VDG did before creating train embeddings.
    #  Cosine similarity threshold?
    # issue: lower casing everything means embeddings only found for lowercase word! Just returning lower case candidates as otherwise losing information
    # AS IN MONOISE
    # can use twitter embeddings from van der Goot - based on distributional hypothesis to find tokens with similar semantics
    # could use cosine similarity as a feature for selection? Using here to get most similar candidates.
    # ISSUE: antonyms also often present in same contexts.
    # done need from_word_embeddings as colinear with embeddings_rank
    candidates = pd.DataFrame(columns=["cosine_to_orig", "embeddings_rank"])
    cands = []
    if tok in vectors:
        cands = [
            c
            for c in vectors.similar_by_vector(tok)
            if is_eligible(c[0]) and c[1] >= threshold and c[0].islower()
        ]
    for rank, c in enumerate(cands):
        k, v = c
        candidates.loc[k] = {
            "cosine_to_orig": v,
            "from_word_embeddings": 1,
            "embeddings_rank": rank,
        }
    return candidates


def norm_lookup(tok, normalisations):
    # TODO: external norm dicts?
    # MONOISE
    # lookup in list of all replacement pairs found in the training data (and external sources?)
    # all norm tokens with raw token tok are included as candidates
    # don't need from_lookup as this is colinear with norms_seen
    candidates = pd.DataFrame(columns=["norms_seen"])
    for k, v in normalisations.get(tok, {}).items():
        candidates.loc[k] = {"norms_seen": v}
    return candidates


def clipping(tok, lex):
    # MONOISE
    # all words in lexicon that have tok as a prefix (capturing abbreviation). May only consider for tok length above 2?
    candidates = pd.DataFrame(columns=["from_clipping"])
    if len(tok) < 2:
        return candidates
    # TODO: length threshold? prune generated (only some degree of clipping allowed w.r.t. edit distance)?
    for c in lex:
        if c.startswith(tok):
            candidates.loc[c] = {"from_clipping": 1}
    return candidates


def split(tok, lex):
    # MONOISE
    # hypothesis splits on (every/some) position and check if both words are in lexicon. May only consider of tok length above 3?
    candidates = pd.DataFrame(columns=["from_split"])
    if len(tok) < 3:
        return candidates
    for pos in range(1, len(tok)):
        left = tok[:pos]
        right = tok[pos:]
        if left in lex and right in lex:
            candidates.loc[" ".join([left, right])] = {"from_split": 1}
    # TODO: recursive candidate generation on each left and right? Probably not... More than one split? Probably not either...
    # TODO: length threshold?
    return candidates


def spellcheck(tok, dictionary):
    # TODO: no control over this - can I change in source code? To load in custom lexicon must completely reimplement!!
    # TODO: rank may have absolutely no meaning here...
    # don't need from_spellcheck as colinear with spellcheck_rank
    candidates = pd.DataFrame(columns=["spellcheck_rank"])
    rank = 0
    for c in dictionary.suggest(tok):
        if c.islower():
            candidates.loc[c] = {"from_spellcheck": 1, "spellcheck_rank": rank}
            rank += 1
    return candidates


if __name__ == "__main__":
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
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
