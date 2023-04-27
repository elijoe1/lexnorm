import os

import numpy as np
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


def add_ngram_chunks(dataframe, ngram_file, col_name, verbose=False):
    dataframe[col_name] = np.nan
    for chunk in pd.read_csv(
        ngram_file,
        index_col=0,
        na_values="",
        header=0,
        names=[col_name],
        dtype={col_name: int},
        keep_default_na=False,
        chunksize=10**6,
    ):
        dataframe[col_name] = dataframe[col_name].fillna(chunk[col_name])
        if verbose:
            print(dataframe[col_name].isna().sum())
    return dataframe


if __name__ == "__main__":
    tsv_to_csv(
        os.path.join(DATA_PATH, "interim/twitter_ngrams.1"),
        os.path.join(DATA_PATH, "processed/twitter_unigrams.ngr"),
    )
    # tsv_to_csv(
    #     os.path.join(DATA_PATH, "interim/twitter_ngrams.2"),
    #     os.path.join(DATA_PATH, "processed/twitter_bigrams.ngr"),
    # )
    tsv_to_csv(
        os.path.join(DATA_PATH, "interim/wiki_ngrams.1"),
        os.path.join(DATA_PATH, "processed/wiki_unigrams.ngr"),
    )
    tsv_to_csv(
        os.path.join(DATA_PATH, "interim/wiki_ngrams.2"),
        os.path.join(DATA_PATH, "processed/wiki_bigrams.ngr"),
    )
