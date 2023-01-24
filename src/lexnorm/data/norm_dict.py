import os
from collections import Counter

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH


def from_train(train) -> dict[str, Counter]:
    # TODO: write test?
    raw, norm = loadNormData(train)
    norm_dict = {}
    for tweet_raw, tweet_norm in zip(raw, norm):
        to_add = []
        for pos, toks in enumerate(zip(tweet_raw, tweet_norm)):
            tok_raw, tok_norm = toks
            if tok_raw != tok_norm:
                if not tok_norm:
                    if not to_add:
                        to_add.append([tweet_raw[pos - 1], tweet_norm[pos - 1]])
                    to_add[-1][0] = " ".join([to_add[-1][0], tok_raw])
                else:
                    to_add.append([tok_raw, tok_norm])
        for pair in to_add:
            norm_dict.setdefault(pair[0], Counter()).update([pair[1]])
    return {k: dict(v) for k, v in norm_dict.items()}


if __name__ == "__main__":
    print(dict(from_train(os.path.join(DATA_PATH, "interim/train.txt"))))
