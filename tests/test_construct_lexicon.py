from lexnorm.data.construct_lexicon import build_lexicon
from lexnorm.definitions import DATA_PATH
import os


def test_build_lexicon():
    lex1 = build_lexicon(
        {"english", "american"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/american-70.txt")) as lexicon:
        for line in lexicon:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference

    lex1 = build_lexicon(
        {"english", "australian"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
        95,
        3,
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/australian-95.txt")) as lexicon:
        for line in lexicon:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference

    lex1 = build_lexicon(
        {"english", "british"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
        35,
        1,
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/british-35.txt")) as lexicon:
        for line in lexicon:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference
