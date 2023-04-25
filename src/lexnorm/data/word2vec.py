import os

import gensim.models

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH


def get_vectors(raw, size: int = 100):
    if size not in [100, 400]:
        raise ValueError(f"vector size must be either 100 or 400, not {size}")
    train_vectors = gensim.models.Word2Vec(
        sentences=raw, vector_size=size, window=5, sg=1, min_count=5, workers=-1
    ).wv
    vdg_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(DATA_PATH, f"external/monoise_data/w2v_{size}.bin"),
        binary=True,
        unicode_errors="ignore",
    )
    vdg_vectors.add_vectors(train_vectors.index_to_key, train_vectors.vectors)
    return vdg_vectors


if __name__ == "__main__":
    raw, _ = loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    vectors = get_vectors(raw, 100)
    print(vectors.most_similar("lmaoo"))
