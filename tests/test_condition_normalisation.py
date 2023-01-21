from lexnorm.evaluation import condition_normalisation
from collections import Counter
from lexnorm.definitions import DATA_PATH
from lexnorm.data import normEval
from lexnorm.models import annotation
import os


def test_correlation():
    # example from wikipedia
    a = Counter(a=3, b=3)
    b = Counter(a=2)
    c = Counter(a=1, b=0)
    d = Counter(a=2, f=1)
    assert round(condition_normalisation.correlation(a, b, c, d), 3) == 0.478


def test_contingency():
    # TODO: make more rigorous e.g. by supplying custom raw and norm
    full_raw, full_norm = normEval.loadNormData(
        os.path.join(DATA_PATH, "interim/train.txt")
    )
    a, b, c, d = condition_normalisation.contingency(
        full_raw, full_norm, lambda x: x[0] == "a", True
    )
    raw_count = 0
    cond_count = 0
    a_count = 0
    c_count = 0
    for tweet_raw, tweet_norm in zip(full_raw, full_norm):
        eligible_list = annotation.list_eligible(tweet_raw)
        for tok_raw, tok_norm, elig in zip(tweet_raw, tweet_norm, eligible_list):
            if elig:
                raw_count += 1
                if tok_raw[0] == "a":
                    cond_count += 1
                    if tok_raw == tok_norm:
                        a_count += 1
                elif tok_raw == tok_norm:
                    c_count += 1
    a_sum = sum(a.values())
    b_sum = sum(b.values())
    c_sum = sum(c.values())
    d_sum = sum(d.values())
    assert a_sum + b_sum + c_sum + d_sum == raw_count
    assert a_sum + b_sum == cond_count
    assert a_sum == a_count
    assert c_sum == c_count
