import math
import os
from collections import Counter

from lexnorm.data import norm_dict
from lexnorm.definitions import DATA_PATH

result_types = {"pair", "raw", "normed"}


# def contingency(
#     raw_list: list[list[str]],
#     norm_list: list[list[str]],
#     condition: Callable[[str], bool],
#     pair: bool = False,
#     norm: bool = False,
# ):
#     """
#     Returns four counters a, b, c, d, representing: a - condition met, not normalised; b - condition met, normalised;
#     c - condition not met, not normalised; d - condition not met, normalised, over all eligible tokens as defined by
#     annotation.list_eligible
#
#     @param raw_list: list of raw tokens
#     @param norm_list: list of corresponding normed phrases
#     @param condition: condition to evaluate
#     @param pair: if counters should count only the token/phrase in the tuple being evaluated, or the whole thing
#     @param norm: if the normed phrase should be evaluated instead of the raw token
#     @return: a tuple of four counters, representing the contingency table's top row |a  b| and bottom |c  d|
#     """
#     p_normed = Counter()
#     p_unnormed = Counter()
#     n_normed = Counter()
#     n_unnormed = Counter()
#     for tweet_raw, tweet_norm in zip(raw_list, norm_list):
#         for raw, normed in zip(tweet_raw, tweet_norm):
#             if is_eligible(raw):
#                 tok = normed if norm else raw
#                 to_update = (raw, normed) if pair else tok
#                 if raw != normed and condition(tok):
#                     p_normed.update([to_update])
#                 elif raw == normed and condition(tok):
#                     p_unnormed.update([to_update])
#                 elif raw != normed and not condition(tok):
#                     n_normed.update([to_update])
#                 else:
#                     n_unnormed.update([to_update])
#     return p_unnormed, p_normed, n_unnormed, n_normed


def contingency_from_dict(norm_dict, condition, result="pair"):
    """
    Produces 2x2 contingency table wrt normalisation and a condition over all normalisation pairs, as encoded in a
    normalisation dictionary.

    :param norm_dict: normalisation dictionary as produced by data.norm_dict
    :param condition: condition on entire normalisation pair
    :param result: which component of the pairs to count - the whole 'pair', the 'raw', or the 'normed' component only
    :return: tuple of four counters a, b, c, d, representing: a - condition met, not normalised; b - condition met, normalised;
    c - condition not met, not normalised; d - condition not met, normalised.
    """
    p_normed = Counter()
    p_unnormed = Counter()
    n_normed = Counter()
    n_unnormed = Counter()
    if result not in result_types:
        raise ValueError("result must be one of 'pair', 'raw', 'normed'")
    for raw in norm_dict.keys():
        for norm in norm_dict[raw].keys():
            count = norm_dict[raw][norm]
            pair = (raw, norm)
            to_update = (
                pair if result == "pair" else (pair[0] if result == "raw" else pair[1])
            )
            if pair[0] == pair[1]:
                if condition(pair):
                    p_unnormed.update([to_update] * count)
                else:
                    n_unnormed.update([to_update] * count)
            else:
                if condition(pair):
                    p_normed.update([to_update] * count)
                else:
                    n_normed.update([to_update] * count)
    return p_unnormed, p_normed, n_unnormed, n_normed


def correlation(
    p_unnormed: Counter, p_normed: Counter, n_unnormed: Counter, n_normed: Counter
) -> float:
    """
    Calculates the phi coefficient between a condition and normalisation from the counters returned by contingency_from_dict.

    :param p_unnormed: counter for unnormalised tokens meeting condition
    :param p_normed: counter for normalised tokens meeting condition
    :param n_unnormed: counter for unnormalised tokens not meeting condition
    :param n_normed: counter for normalised tokens not meeting condition
    :return: phi (aka matthew's correlation) coefficient between normalisation and the condition
    """
    a = sum(p_unnormed.values())
    b = sum(p_normed.values())
    c = sum(n_unnormed.values())
    d = sum(n_normed.values())
    denom = math.sqrt((a + b) * (b + d) * (a + c) * (c + d))
    # correct limiting value according to wikipedia
    denom = denom if denom else 1
    phi = (a * d - b * c) / denom
    return phi
