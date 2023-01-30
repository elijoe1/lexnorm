from lexnorm.models.filtering import is_eligible

# TODO exclusion of modules capability?
def generate_candidates(tweet: list[str]) -> list[set[str]]:
    """
    Given an input tweet, produce a list of normalisation candidates for each token.

    # TODO ability to exclude specific modules/steps?
    # TODO implement generation for eligible tokens, not forgetting many-to-one normalisations

    :param tweet: to normalise, as list of tokens
    :return: a corresponding list of sets of normalisation candidates
    """
    candidates = []
    for i, tok in enumerate(tweet):
        if tok == "rt":
            # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
            # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
            # of tweet and not followed by @mention) and when normalised, always to 'retweet'
            if 0 < i < len(tweet) - 1 and tweet[i + 1][0] != "@":
                candidates.append({"retweet"})
            else:
                candidates.append({tok})
        elif is_eligible(tok):
            ...
        else:
            candidates.append({tok})
    return candidates
