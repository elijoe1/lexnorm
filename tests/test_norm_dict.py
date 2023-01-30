import os

from lexnorm.data import baseline, norm_dict


def test_construct(tmp_path):
    raw = [
        ["bruther", "get", "outt", "youre", "feelins", "loll"],
        ["my", "brother", "thinkgs", "youre", "trippin", "loll"],
    ]
    norm = [
        ["brother", "get", "out", "your", "feelings", "lol"],
        ["my", "brother", "thinks", "you're", "tripping", "lol"],
    ]
    baseline.write(raw, norm, os.path.join(tmp_path, "tmp"))
    normalisations = norm_dict.construct(os.path.join(tmp_path, "tmp"))
    assert normalisations == {
        "bruther": {"brother": 1},
        "get": {"get": 1},
        "outt": {"out": 1},
        "youre": {"your": 1, "you're": 1},
        "feelins": {"feelings": 1},
        "loll": {"lol": 2},
        "my": {"my": 1},
        "brother": {"brother": 1},
        "thinkgs": {"thinks": 1},
        "trippin": {"tripping": 1},
    }
