import os

import pandas as pd

from lexnorm.definitions import DATA_PATH


def tsv_to_csv(ngram_path, output_path):
    with open(ngram_path) as f:
        ngrams = pd.read_csv(
            f,
            sep="\t",
            index_col=0,
            names=["frequency"],
            na_values="",
            keep_default_na=False,
        )
    lowercase = ngrams.groupby(ngrams.index.str.lower()).sum()
    with open(output_path, "w") as f:
        lowercase.to_csv(f)


def ngram_from_csv(df_path):
    with open(df_path, "r") as f:
        df = pd.read_csv(f, index_col=0, na_values="", keep_default_na=False)
    return df


if __name__ == "__main__":
    # tsv_to_csv(
    #     os.path.join(DATA_PATH, "interim/twitter_ngrams.1"),
    #     os.path.join(DATA_PATH, "processed/twitter_unigrams.ngr"),
    # )
    tsv_to_csv(
        os.path.join(DATA_PATH, "interim/twitter_ngrams.2"),
        os.path.join(DATA_PATH, "processed/twitter_bigrams.ngr"),
    )
    # tsv_to_csv(
    #     os.path.join(DATA_PATH, "interim/wiki_ngrams.1"),
    #     os.path.join(DATA_PATH, "processed/wiki_unigrams.ngr"),
    # )
    # tsv_to_csv(
    #     os.path.join(DATA_PATH, "interim/wiki_ngrams.2"),
    #     os.path.join(DATA_PATH, "processed/wiki_bigrams.ngr"),
    # )
