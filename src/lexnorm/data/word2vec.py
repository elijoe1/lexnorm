import os

import gensim.models

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH


def get_vectors(path):
    raw, _ = loadNormData(path)
    model = gensim.models.Word2Vec(sentences=raw)
    return model.wv


if __name__ == "__main__":
    get_vectors(os.path.join(DATA_PATH, "interim/train.txt"))
    # new_model = gensim.models.KeyedVectors.load_word2vec_format(
    #     os.path.join(DATA_PATH, "external/monoise_data/w2v.bin"),
    #     binary=True,
    #     unicode_errors="ignore",
    # )
