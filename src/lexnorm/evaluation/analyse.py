from lexnorm.generate_extract.filtering import is_eligible


def analyse(raw, norm):
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
                elif is_eligible(raw_tok):
                    normalisations += 1
    return eligible, normalisations


def get_tokens_from_ids(ids, raw, norm):
    id = -1
    tokens = []
    for raw_tweet, norm_tweet in zip(raw, norm):
        for raw_tok, norm_tok in zip(raw_tweet, norm_tweet):
            if is_eligible(raw_tok):
                id += 1
                if id in ids:
                    tokens.append((raw_tok, norm_tok))
    return tokens
