import os
import pickle
from collections import Counter

from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.filtering import is_eligible


def binary_to_pickle(ngram_path, output_path):
    ngrams = Counter()
    with open(ngram_path) as f:
        for line in f:
            ngram = line.strip().split("\t")
            if len(ngram) != 2:
                continue
            # merge all cased versions of token as everything is lower-cased in our system - no other way to do this?
            ngrams.update({ngram[0].lower(): int(ngram[1])})
    with open(output_path, "wb") as f:
        pickle.dump(ngrams, f)


def counter_from_pickle(counter_path):
    with open(counter_path, "rb") as f:
        counter = pickle.load(f)
    return counter


if __name__ == "__main__":
    binary_to_pickle(
        os.path.join(DATA_PATH, "interim/twitter_ngrams.1"),
        os.path.join(DATA_PATH, "processed/twitter_unigram_counter.pickle"),
    )
    binary_to_pickle(
        os.path.join(DATA_PATH, "interim/twitter_ngrams.2"),
        os.path.join(DATA_PATH, "processed/twitter_bigram_counter.pickle"),
    )
    binary_to_pickle(
        os.path.join(DATA_PATH, "interim/wiki_ngrams.1"),
        os.path.join(DATA_PATH, "processed/wiki_unigram_counter.pickle"),
    )
    binary_to_pickle(
        os.path.join(DATA_PATH, "interim/wiki_ngrams.2"),
        os.path.join(DATA_PATH, "processed/wiki_bigram_counter.pickle"),
    )
    # counter = counter_from_pickle(
    #     os.path.join(DATA_PATH, "processed/wiki_bigram_counter.pickle")
    # )
    # print(counter.most_common(40))
