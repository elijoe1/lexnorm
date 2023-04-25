import math
from collections import Counter
from lexnorm.data import norm_dict
from lexnorm.generate_extract.filtering import is_eligible

result_types = {"pair", "raw", "normed"}


def contingency(raw, norm, condition, result="pair", eligible_only=True):
    """
    Wrapper for contingency_from_dict allowing input of tweets directly rather than derived normalisation dictionary.

    :param raw: raw tweets
    :param norm: normed tweets
    :param condition: condition over normalisation pair
    :param result: which component of the pairs to count - the whole 'pair', the 'raw', or the 'normed' component only
    :param eligible_only: only consider over eligible raw tokens
    :return: tuple of four counters a, b, c, d, representing: a - condition met, not normalised; b - condition met, normalised;
    c - condition not met, not normalised; d - condition not met, normalised.
    """
    normalisations = norm_dict.construct(raw, norm, exclude_mto=False)
    return contingency_from_dict(normalisations, condition, result, eligible_only)


def contingency_from_dict(norm_dict, condition, result="pair", eligible_only=True):
    """
    Produces 2x2 contingency table wrt normalisation and some condition over normalisation pairs, as encoded in a
    normalisation dictionary.

    :param norm_dict: normalisation dictionary as produced by data.norm_dict
    :param condition: condition over normalisation pair
    :param result: which component of the pairs to count - the whole 'pair', the 'raw', or the 'normed' component only
    :param eligible_only: only consider over eligible raw tokens
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
        if not eligible_only or is_eligible(raw):
            for norm in norm_dict[raw].keys():
                count = norm_dict[raw][norm]
                pair = (raw, norm)
                to_update = (
                    pair
                    if result == "pair"
                    else (pair[0] if result == "raw" else pair[1])
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
