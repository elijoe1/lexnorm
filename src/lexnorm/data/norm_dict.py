import os
from collections import Counter

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.filtering import is_eligible


def construct(raw, norm):
    """
    Creates a dictionary of normalisations seen in a dataset.

    For each eligible raw token, creates a dictionary of all normalisations seen (including no normalisation) and the frequency
    of each. Note many-to-one normalisations are stored properly, with the raw phrase being all the involved raw tokens
    joined by spaces.

    :param raw: raw tweets
    :param norm: norm tweets
    :return: Dictionary of {raw -> {norm -> freq}}
    """
    norm_dict = {}
    for tweet_raw, tweet_norm in zip(raw, norm):
        to_add = []
        for tok_raw, tok_norm in zip(tweet_raw, tweet_norm):
            if not is_eligible(tok_raw):
                continue
            if not tok_norm:
                # handle many-to-one normalisations
                to_add[-1][0] = " ".join([to_add[-1][0], tok_raw])
            else:
                to_add.append([tok_raw, tok_norm])
        for pair in to_add:
            norm_dict.setdefault(pair[0], Counter()).update([pair[1]])
    return {k: dict(v) for k, v in norm_dict.items()}


if __name__ == "__main__":
    raw, norm = loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    print(construct(raw, norm))
