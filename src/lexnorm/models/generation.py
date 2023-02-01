import os
import pickle
import time

from lexnorm.data import word2vec, norm_dict, normEval
from lexnorm.definitions import DATA_PATH
from lexnorm.models.filtering import is_eligible
from spylls.hunspell import Dictionary
import pandas as pd
import multiprocessing as mp


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

# def


def candidates_from_token(
    orig, vectors, normalisations, lexicon, spellcheck_dictionary
):
    candidates = original_token(orig)
    candidates = candidates.combine_first(word_embeddings(orig, vectors))
    candidates = candidates.combine_first(norm_lookup(orig, normalisations))
    candidates = candidates.combine_first(spellcheck(orig, spellcheck_dictionary))
    # # obviously lookup on the train set will always produce the correct candidate (perhaps among others)!
    candidates = candidates.combine_first(clipping(orig, lexicon))
    candidates = candidates.combine_first(split(orig, lexicon))
    return candidates


def original_token(tok):
    # AS IN MONOISE
    # needed if detect step is skipped, as all tokens will be replaced by one from the list of candidates
    candidate = pd.DataFrame(columns=["cosine_to_orig", "from_original_token"])
    candidate.loc[tok] = {"from_original_token": 1, "cosine_to_orig": 1}
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
    candidates = pd.DataFrame(columns=["cosine_to_orig", "from_word_embeddings"])
    cands = []
    if tok in vectors:
        cands = [
            c
            for c in vectors.similar_by_vector(tok)
            if is_eligible(c[0]) and c[1] >= threshold and c[0].islower()
        ]
    for k, v in cands:
        candidates.loc[k] = {"cosine_to_orig": v, "from_word_embeddings": 1}
    return candidates


def norm_lookup(tok, normalisations):
    # TODO: external norm dicts?
    # MONOISE
    # lookup in list of all replacement pairs found in the training data (and external sources?)
    # all norm tokens with raw token tok are included as candidates
    candidates = pd.DataFrame(columns=["norms_seen", "from_norm_lookup"])
    for k, v in normalisations.get(tok, {}).items():
        candidates.loc[k] = {"norms_seen": v, "from_norm_lookup": 1}
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
    candidates = pd.DataFrame(columns=["from_spellcheck"])
    for c in dictionary.suggest(tok):
        if c.islower():
            candidates.loc[c] = {"from_spellcheck": 1}
    return candidates


if __name__ == "__main__":
    # TODO use multithreading for generate_candidates for speedup. (possible as all data used is read only).
    # to do this may need to load all the data for each process (ACTUALLY think shared memory is fine...investigate)
    # make a bunch of processes each loading everything, and processing some amount of tweets. spawn all of them
    # and give them each some amount of tweets. must tell slurm how many cores want to use.
    # can point to files in rds, then each process can load this into its ram. Each core has about 4gb.
    # could do: master script spawns 50 processes, each does df for 10 tweets and returns this. we join this all together
    # and append to csv and repeat (to prevent overfilling memory).
    # module load python, create venv. Then in script, activate venv, install requirements, run script. files in rds
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "interim/train.txt"))
    w2v = word2vec.get_vectors(os.path.join(DATA_PATH, "interim/train.txt"))
    with open(os.path.join(DATA_PATH, "interim/lexicon.txt"), "rb") as lf:
        lex = pickle.load(lf)
    normalisations = norm_dict.construct(os.path.join(DATA_PATH, "interim/train.txt"))
    spellcheck_dict = Dictionary.from_files("en_US")
    # threads = []
    start = time.time()
    train_data = pd.DataFrame()
    for tweet in raw[:10]:
        for tok in tweet:
            train_data = pd.concat(
                [
                    train_data,
                    candidates_from_token(
                        tok, w2v, normalisations, lex, spellcheck_dict
                    ),
                ]
            )
    # for i in range(10):
    #     t = threading.Thread(
    #         target=generate_candidates,
    #         args=(raw[i : i + 10], w2v, lex, normalisations, spellcheck_dict),
    #     )
    #     threads.append(t)
    #     t.start()
    # for t in threads:
    #     t.join()
    end = time.time()
    print(end - start)
    # about 40 secs not multithreaded for 10 tweets - 36 hours overall (!)
    # 25 tweets in 80 secs! better. 366 seconds for 100 tweets -
    with open(os.path.join(DATA_PATH, "interim/candidates.txt"), "w") as f:
        train_data.to_csv(f)
