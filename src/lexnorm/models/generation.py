import json
import os
import pickle
import time

from lexnorm.data import word2vec, norm_dict, lexicon, normEval
from lexnorm.definitions import DATA_PATH
from lexnorm.models.filtering import is_eligible
from spylls.hunspell import Dictionary
import pandas as pd

# TODO exclusion of modules capability?
def generate_candidates(
    tweet: list[str], vectors, lexicon, normalisations, spellcheck_dict
) -> list[list[str]]:
    """
    Given an input tweet, produce a list of normalisation candidates for each token.

    # TODO ability to exclude specific modules/steps?
    # TODO implement generation for eligible tokens, not forgetting many-to-one normalisations
    if we have no contextual features, we can generate candidates independently for each token - for each we can
    extract features, then put all in a massive dataframe, labelling as correct or incorrect and train!!! So this
    function has to, for a given tweet, produce all the candidates and their features per token. The candidate features
    ust include the features of the original word as this is how we decide to normalise, and must be based on comparing
    the candidate and the original token to evaluate how realistic normalising to is.

    :param normalisations: normalisation dictionary for the normalisation lookup module
    :param lexicon: lexicon for the split and clipping modules
    :param vectors: vectors for the embeddings lookup module
    :param tweet: to normalise, as list of tokens
    :return: a corresponding list of sets of normalisation candidates
    """
    candidates_list = []
    for i, tok in enumerate(tweet):
        if tok == "rt":
            # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
            # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
            # of tweet and not followed by @mention) and when normalised, always to 'retweet'
            if 0 < i < len(tweet) - 1 and tweet[i + 1][0] != "@":
                candidates_list.append(["retweet"])
            else:
                candidates_list.append([tok])
        elif is_eligible(tok):
            candidates = pd.DataFrame()
            candidates = set()
            candidates = candidates.union(original_token(tok))
            candidates = candidates.union(word_embeddings(tok, vectors))
            candidates = candidates.union(spellcheck(tok, spellcheck_dict))
            # obviously lookup on the train set will always produce the correct candidate (perhaps among others)!
            candidates = candidates.union(lookup(tok, normalisations))
            candidates = candidates.union(clipping(tok, lexicon))
            candidates = candidates.union(split(tok, lexicon))
            candidates_list.append(list(candidates))
        else:
            candidates_list.append([tok])
    return candidates_list


def original_token(tok):
    # AS IN MONOISE
    # needed if detect step is skipped, as all tokens will be replaced by one from the list of candidates
    return {tok}


def word_embeddings(tok, vectors, threshold=0):
    # TODO uni, bigram freqs?
    # TODO implement word2vec with keras. Experiment with different no. of candidates generated.
    #  Could even create twitter embeddings myself? Could clean up as VDG did before creating train embeddings.
    #  Cosine similarity threshold?
    # AS IN MONOISE
    # can use twitter embeddings from van der Goot - based on distributional hypothesis to find tokens with similar semantics
    # could use cosine similarity as a feature for selection? Using here to get most similar candidates.
    # ISSUE: antonyms also often present in same contexts.
    candidates = set()
    if tok in vectors:
        candidates = set(vectors.similar_by_vector(tok))
    return {c[0].lower() for c in candidates if is_eligible(c[0]) and c[1] >= threshold}


def lookup(tok, dictionary):
    # TODO: external norm dicts?
    # MONOISE
    # lookup in list of all replacement pairs found in the training data (and external sources?)
    # all norm tokens with raw token tok are included as candidates
    return {v for v in dictionary.get(tok, {}).keys()}


def clipping(tok, lex):
    # MONOISE
    # all words in lexicon that have tok as a prefix (capturing abbreviation). May only consider for tok length above 2?
    candidates = set()
    if len(tok) < 2:
        return set()
    # TODO: length threshold? prune generated (only some degree of clipping allowed w.r.t. edit distance)?
    return [t for t in lex if t.startswith(tok)]


def split(tok, lex):
    # MONOISE
    # hypothesis splits on (every/some) position and check if both words are in lexicon. May only consider of tok length above 3?
    candidates = set()
    if len(tok) < 3:
        return set()
    for pos in range(1, len(tok)):
        left = tok[:pos]
        right = tok[pos:]
        if left in lex and right in lex:
            candidates.add(" ".join([left, right]))
    # TODO: recursive candidate generation on each left and right? Probably not... More than one split? Probably not either...
    # TODO: length threshold?
    return candidates


def spellcheck(tok, dictionary):
    # TODO: no control over this - can I change in source code? To load in custom lexicon must completely reimplement!!
    return {c.lower() for c in dictionary.suggest(tok)}


if __name__ == "__main__":
    # TODO use multithreading for generate_candidates for speedup. (possible as all data used is read only).
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "interim/train.txt"))
    w2v = word2vec.get_vectors(os.path.join(DATA_PATH, "interim/train.txt"))
    with open(os.path.join(DATA_PATH, "interim/lexicon.txt"), "rb") as lf:
        lex = pickle.load(lf)
    normalisations = norm_dict.construct(os.path.join(DATA_PATH, "interim/train.txt"))
    spellcheck_dict = Dictionary.from_files("en_US")
    start = time.time()
    cands = []
    for raw_tweet, _ in zip(raw[:10], norm[:10]):
        cands_tweet = generate_candidates(
            raw_tweet, w2v, lex, normalisations, spellcheck_dict
        )
        cands.append(cands_tweet)
    end = time.time()
    print(end - start)
    with open(os.path.join(DATA_PATH, "interim/candidates.txt"), "w") as f:
        json.dump(cands, f)
