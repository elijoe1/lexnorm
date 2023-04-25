import os
from collections import Counter

from lexnorm.data.normEval import loadNormData
from lexnorm.definitions import DATA_PATH


def construct(raw, norm, exclude_mto=True):
    """
    Creates a dictionary of normalisations seen in a dataset.

    For each raw token, creates a dictionary of all normalisations seen (including no normalisation) and the frequency
    of each. Note many-to-one normalisations are not included by default, as using the normalisation for any of the raw tokens
    individually or including a multi-token raw phrase doesn't make sense for candidate generation (done tokenwise).
    If included (for analysis), each involved raw token will be included separately.

    :param raw: raw tweets
    :param norm: norm tweets
    :param exclude_mto: whether to include many-to-one normalisations
    :return: Dictionary of {raw -> {norm -> freq}}
    """
    norm_dict = {}
    for tweet_raw, tweet_norm in zip(raw, norm):
        to_add = []
        for tok_raw, tok_norm in zip(tweet_raw, tweet_norm):
            if not tok_norm and exclude_mto:
                # capture many-to-one normalisations
                to_add[-1][0] = " ".join([to_add[-1][0], tok_raw])
            else:
                to_add.append([tok_raw, tok_norm])
        for pair in to_add:
            if " " in pair[0]:
                # detect and disregard many-to-one normalisations
                continue
            norm_dict.setdefault(pair[0], Counter()).update([pair[1]])
    return {k: dict(v) for k, v in norm_dict.items()}


if __name__ == "__main__":
    raw, norm = loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    print(construct(raw, norm)["wat"])
