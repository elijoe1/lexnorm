from lexnorm.data.normEval import loadNormData
from lexnorm.generate_extract.filtering import is_eligible


def analyse(data_path):
    raw, norm = loadNormData(data_path)
    eligible = 0
    normalisations = 0
    for raw_tweet, norm_tweet in zip(raw, norm):
        for raw_tok, norm_tok in zip(raw_tweet, norm_tweet):
            if is_eligible(raw_tok):
                eligible += 1
            if raw_tok != norm_tok:
                if not is_eligible(raw_tok) and raw_tok != "rt":
                    print(raw_tok)
                    exit(-1)
                else:
                    normalisations += 1
    return eligible, normalisations
