import os
import pickle
from collections import Counter

from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.filtering import is_eligible


def load_ngrams_as_counter(ngram_path, output_path):
    ngrams = Counter()
    with open(ngram_path) as f:
        for line in f:
            ngram = line.strip().split("\t")
            if len(ngram) < 2:
                continue
            elif is_eligible(ngram[0]):
                ngrams.update({ngram[0]: int(ngram[1])})
    with open(output_path, "wb") as f:
        pickle.dump(ngrams, f)


def load_ngram_counter(counter_path):
    with open(counter_path, "rb") as f:
        counter = pickle.load(f)
    return counter


if __name__ == "__main__":
    load_ngrams_as_counter(
        os.path.join(DATA_PATH, "interim/twitter_ngrams_old.1"),
        os.path.join(DATA_PATH, "processed/twitter_unigram_counter_old"),
    )
    load_ngrams_as_counter(
        os.path.join(DATA_PATH, "interim/twitter_ngrams_old.2"),
        os.path.join(DATA_PATH, "processed/twitter_bigram_counter_old"),
    )
    load_ngrams_as_counter(
        os.path.join(DATA_PATH, "interim/wiki_ngrams_old.1"),
        os.path.join(DATA_PATH, "processed/wiki_unigram_counter_old"),
    )
    load_ngrams_as_counter(
        os.path.join(DATA_PATH, "interim/wiki_ngrams_old.2"),
        os.path.join(DATA_PATH, "processed/wiki_bigram_counter_old"),
    )
