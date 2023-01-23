from lexnorm.definitions import DATA_PATH
from lexnorm.data import lexicon
import os


def test_build():
    lex1 = lexicon.build(
        {"english", "american"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
        profane=False,
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/american-70.txt")) as lex:
        for line in lex:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference

    lex1 = lexicon.build(
        {"english", "australian"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
        95,
        3,
        profane=False,
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/australian-95.txt")) as lex:
        for line in lex:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference

    lex1 = lexicon.build(
        {"english", "british"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
        35,
        1,
        profane=False,
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "test/british-35.txt")) as lex:
        for line in lex:
            lex2.add(line.strip().lower())
    difference = lex1 ^ lex2
    assert not difference


# def test_
