# def list_eligible(tweet: list[str]) -> list[bool]:
#     """
#
#     @param tweet: list of tokens to investigate
#     @return: corresponding boolean list giving eligibility for normalisation
#     under W-NUT 2015 annotation guidelines
#     """
#     eligible_list = []
#     for i, token in enumerate(tweet):
#         # check if contains forbidden characters (anything but numbers, letters, and internal apostrophes)
#         # as described in 2015 guideline 3 (interpreting apostrophe 'used in contraction' vs 'as single quote'
#         # as meaning internal vs external).
#         if (
#             any(not (char.isalnum() or char == "'") for char in token)
#             or token[0] == "'"
#             or token[-1] == "'"
#         ):
#             eligible_list.append(False)
#             continue
#         # check if domain specific 'rt' as described in notebook 1.0 and 2015 guideline 3
#         if token == "rt":
#             if i == 0:
#                 eligible_list.append(False)
#                 continue
#             elif 0 < i < len(tweet) - 1:
#                 eligible_list.append(False if tweet[i + 1][0] == "@" else True)
#                 continue
#         eligible_list.append(True)
#     return eligible_list


def is_eligible(tok: str, allow_special: bool = False) -> bool:
    """
    Determines if a token is eligible to be normalised under the dataset guidelines.

    Checks if token contains forbidden characters (anything but numbers, letters, spaces, and internal apostrophes)
    as described in 2015 guideline 3 (interpreting apostrophes 'used in contraction' vs 'as single quote' as internal
    vs external respectively). Also checks if token is 'rt' (domain specific entity handled specially during generation).
    Allows whitespace as this is never present in the raw data, but allows for components of normalisation pairs
    for one-to-many and many-to-one normalisations to be seen as eligible.

    Under this definition, all raw and normalised phrases in the train and dev sets are eligible, as desired.

    :param allow_special: if the special tokens <S>, </S> should be allowed (used in ngrams from VDG)
    :param tok: token to check
    :return: if eligible for normalisation/as a candidate
    """
    if allow_special and tok in ["<S>", "<s>", "</S>", "</s>"]:
        return True
    if allow_special and tok.startswith(("<S>", "<s>")):
        return is_eligible(tok[3:])
    elif allow_special and tok.endswith(("</S>", "</s>")):
        return is_eligible(tok[:-4])
    elif (
        any(
            not (
                # ascii check needed to prevent chinese characters and such which are considered alphabetic
                (char.isalnum() and char.isascii())
                or char == "'"
                or char == " "
            )
            for char in tok
        )
        or tok[0] == "'"
        or tok[-1] == "'"
    ):
        return False
    elif tok == "rt":
        return False
    else:
        return True
