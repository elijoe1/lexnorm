from lexnorm.evaluation import condition_normalisation
from collections import Counter
from lexnorm.data import baseline
from lexnorm.data import norm_dict
import os


def test_correlation():
    # example from wikipedia
    a = Counter(a=3, b=3)
    b = Counter(a=2)
    c = Counter(a=1, b=0)
    d = Counter(a=2, f=1)
    assert round(condition_normalisation.correlation(a, b, c, d), 3) == 0.478
    assert condition_normalisation.correlation(Counter(), Counter(), c, d) == 0


def test_contingency_from_dict(tmp_path):
    raw = [
        ["bruther", "get", "outt", "youre", "feelins", "loll"],
        ["my", "brother", "thinkgs", "youre", "trippin"],
    ]
    norm = [
        ["brother", "get", "out", "your", "feelings", "lol"],
        ["my", "brother", "thinks", "you're", "tripping"],
    ]
    normalisations = norm_dict.construct(raw, norm)
    a, b, c, d = condition_normalisation.contingency_from_dict(
        normalisations, lambda x: len(x[0]) <= 4
    )
    assert dict(a) == {("my", "my"): 1, ("get", "get"): 1}
    assert dict(b) == {("loll", "lol"): 1, ("outt", "out"): 1}
    assert dict(c) == {("brother", "brother"): 1}
    assert dict(d) == {
        ("bruther", "brother"): 1,
        ("youre", "your"): 1,
        ("feelins", "feelings"): 1,
        ("thinkgs", "thinks"): 1,
        ("trippin", "tripping"): 1,
        ("youre", "you're"): 1,
    }
