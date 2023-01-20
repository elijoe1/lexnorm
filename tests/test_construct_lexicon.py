from lexnorm.data.construct_lexicon import build_lexicon
from lexnorm.definitions import DATA_PATH
import os


def test_build_lexicon():
    lex1 = build_lexicon(
        {"english", "american"},
        {"words", "upper", "proper-names", "contractions", "abbreviations"},
    )
    lex2 = set()
    with open(os.path.join(DATA_PATH, "interim/american-70.txt")) as lexicon:
        for line in lexicon:
            lex2.add(line.strip().lower())
    assert lex1 == lex2
