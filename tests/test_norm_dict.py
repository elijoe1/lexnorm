from lexnorm.data import norm_dict


def test_construct(tmp_path):
    raw = [
        ["bruther", "get", "outt", "youre", "feelins", "loll"],
        ["my", "brother", "thinkgs", "youre", "trippin", "loll", "nut", "case"],
    ]
    norm = [
        ["brother", "get", "out", "your", "feelings", "lol"],
        ["my", "brother", "thinks", "you are", "tripping", "lol", "nutcase", ""],
    ]
    normalisations = norm_dict.construct(raw, norm)
    assert normalisations == {
        "bruther": {"brother": 1},
        "get": {"get": 1},
        "outt": {"out": 1},
        "youre": {"your": 1, "you are": 1},
        "feelins": {"feelings": 1},
        "loll": {"lol": 2},
        "my": {"my": 1},
        "brother": {"brother": 1},
        "thinkgs": {"thinks": 1},
        "trippin": {"tripping": 1},
    }
    normalisations = norm_dict.construct(raw, norm, exclude_mto=False)
    assert normalisations == {
        "bruther": {"brother": 1},
        "get": {"get": 1},
        "outt": {"out": 1},
        "youre": {"your": 1, "you are": 1},
        "feelins": {"feelings": 1},
        "loll": {"lol": 2},
        "my": {"my": 1},
        "brother": {"brother": 1},
        "thinkgs": {"thinks": 1},
        "trippin": {"tripping": 1},
        "nut": {"nutcase": 1},
        "case": {"": 1},
    }
