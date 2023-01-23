def list_eligible(tweet: list[str]) -> list[bool]:
    """

    @param tweet: list of tokens to investigate
    @return: corresponding boolean list giving eligibility for normalisation
    under W-NUT 2015 annotation guidelines
    """
    eligible_list = []
    for i, token in enumerate(tweet):
        # check if contains forbidden characters (anything but numbers, letters, and internal apostrophes)
        # as described in 2015 guideline 3 (interpreting apostrophe 'used in contraction' vs 'as single quote'
        # as meaning internal vs external).
        if (
            any(not (char.isalnum() or char == "'") for char in token)
            or token[0] == "'"
            or token[-1] == "'"
        ):
            eligible_list.append(False)
            continue
        # check if domain specific 'rt' as described in notebook 1.0 and 2015 guideline 3
        if token == "rt":
            if i == 0:
                eligible_list.append(False)
                continue
            elif 0 < i < len(tweet) - 1:
                eligible_list.append(False if tweet[i + 1][0] == "@" else True)
                continue
        eligible_list.append(True)
    return eligible_list
