from collections import Counter
import math
from typing import Callable
from lexnorm.models import annotation


def contingency(
    raw_list: list[list[str]],
    norm_list: list[list[str]],
    condition: Callable[[str], bool],
    pair: bool = False,
    norm: bool = False,
):
    """
    Returns four counters a, b, c, d, representing: a - condition met, not normalised; b - condition met, normalised;
    c - condition not met, not normalised; d - condition not met, normalised, over all eligible tokens as defined by
    annotation.list_eligible

    @param raw_list: list of raw tokens
    @param norm_list: list of corresponding normed phrases
    @param condition: condition to evaluate
    @param pair: if counters should count only the token/phrase in the tuple being evaluated, or the whole thing
    @param norm: if the normed phrase should be evaluated instead of the raw token
    @return: a tuple of four counters, representing the contingency table's top row |a  b| and bottom |c  d|
    """
    p_normed = Counter()
    p_unnormed = Counter()
    n_normed = Counter()
    n_unnormed = Counter()
    for tweet_raw, tweet_norm in zip(raw_list, norm_list):
        eligible_list = annotation.list_eligible(tweet_raw)
        for raw, normed, elig in zip(tweet_raw, tweet_norm, eligible_list):
            if elig:
                tok = normed if norm else raw
                to_update = (tok, normed) if pair else tok
                if raw != normed and condition(tok):
                    p_normed.update([to_update])
                elif raw == normed and condition(tok):
                    p_unnormed.update([to_update])
                elif raw != normed and not condition(tok):
                    n_normed.update([to_update])
                else:
                    n_unnormed.update([to_update])
    return p_unnormed, p_normed, n_unnormed, n_normed


def correlation(
    p_unnormed: Counter, p_normed: Counter, n_unnormed: Counter, n_normed: Counter
) -> float:
    """

    @param p_unnormed: counter for unnormalised tokens meeting condition
    @param p_normed: counter for normalised tokens meeting condition
    @param n_unnormed: counter for unnormalised tokens not meeting condition
    @param n_normed: counter for normalised tokens not meeting condition
    @return: phi (aka matthew's correlation) coefficient between normalisation and the condition
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
