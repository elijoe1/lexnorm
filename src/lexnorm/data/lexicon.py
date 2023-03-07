import os
import pickle

import requests

from lexnorm.data import norm_dict
from lexnorm.definitions import DATA_PATH
from lexnorm.definitions import LEX_PATH
from lexnorm.evaluation import condition_normalisation
from lexnorm.data import normEval

defined_languages = {
    "english",
    "american",
    "australian",
    "canadian",
    "british",
    "british_z",
}
defined_sizes = {10, 20, 35, 40, 50, 55, 60, 70, 80, 95}
defined_variants = {1, 2, 3}
defined_subsets = {"abbreviations", "contractions", "proper-names", "upper", "words"}


def build(
    languages: set[str],
    subsets: set[str],
    max_size: int = 70,
    max_variant: int = 0,
    special: bool = True,
    profane: bool = True,
    lex_path: str = LEX_PATH,
    verbose: bool = False,
) -> set[str]:
    """
    Create (lowercase) word set from SCOWL word lists extending perl script 'mk-list'
    @param languages: including english, american, australian, canadian, british, british_z.
    english should always be included.
    @param subsets: within each language. Including abbreviations, contractions, proper-names, upper, words
    @param max_size: up to 95. 70 is the recommended large size, 50 the medium and 35 the small
    @param max_variant: word variant inclusion level, up to 3 (variants up to level 2 are locale-specific)
    @param profane: if profanity should be included (from misc folder in scowl)
    @param special: if the special word lists (hacker and roman numerals) should be included if the size is large enough
    @param lex_path: path to SCOWL word lists
    @param verbose: if the word lists used should be output
    @return: set containing all words in corresponding word lists
    """
    lexicon = set()
    files = os.listdir(lex_path)
    components = []
    categories = languages.intersection(defined_languages)
    for variant in {v for v in defined_variants if v <= max_variant}:
        for language in {
            f"{l}_" if l != "american" else ""
            for l in languages.intersection(defined_languages) - {"english"}
        }:
            categories.add(f"{language}variant_{variant}")
    if max_variant >= 3:
        categories.add("variant_3")
    for category in categories:
        for subset in subsets.intersection(defined_subsets):
            for size in {s for s in defined_sizes if s <= max_size}:
                to_add = f"{category}-{subset}.{size}"
                if to_add in files:
                    components.append(to_add)
    if special:
        if max_size >= 35:
            components.append("special-roman-numerals.35")
        if max_size >= 50:
            components.append("special-hacker.50")
    if profane:
        components.append("../misc/profane.1")
        components.append("../misc/profane.3")
    if verbose:
        print(f"Component files: {components}")
    for component in components:
        with open(os.path.join(LEX_PATH, component), encoding="iso-8859-1") as file:
            for line in file:
                lexicon.add(line.strip().lower())
    return lexicon


def refine(lexicon: set[str]) -> set[str]:
    filtered_lexicon = set()
    for word in lexicon:
        if len(word) <= 1 and word not in ["a", "i"]:
            continue
        filtered_lexicon.add(word)
    return filtered_lexicon


def build_abbreviations():
    abbrev_lex = set()
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:English_internet_laughter_slang",
        # "cmtitle": "Category:English_internet_slang",
        "cmprop": "title",
        "cmlimit": "max",
        "format": "json",
    }
    response = requests.get("https://en.wiktionary.org/w/api.php", params)
    response.raise_for_status()
    while True:
        response = requests.get("https://en.wiktionary.org/w/api.php", params)
        response.raise_for_status()
        resp = response.json()
        for member in resp["query"]["categorymembers"]:
            if member["title"].lower().isalnum():
                abbrev_lex.add(member["title"].lower())
        if "continue" not in resp.keys():
            break
        else:
            for k, v in resp["continue"].items():
                params[k] = v
    return abbrev_lex


def evaluate(norm_dict, lexicon):
    """
    Outputs statistics for a given lexicon
    """
    a, b, c, d = condition_normalisation.contingency_from_dict(
        norm_dict, lambda x: x[0] in lexicon
    )
    print(
        f"Correlation of in lexicon with normalisation: {condition_normalisation.correlation(a, b, c, d):.2f}"
    )
    print(f"Most common normalised raw phrases in lexicon: {b.most_common(20)}")
    print(f"Most common un-normalised raw phrases not in lexicon: {c.most_common(20)}")
    a, b, _, _ = condition_normalisation.contingency_from_dict(
        norm_dict, lambda x: x[1] not in lexicon
    )
    print(f"Most common token normalisations not in lexicon: {(a + b).most_common(20)}")


if __name__ == "__main__":
    # def powerset(iterable):
    #     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #     s = list(iterable)
    #     return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    # abbrevs = build_abbreviations()
    # max_performance = 0
    # # for languages in list(powerset(defined_languages - {"english"})):
    # # for subsets in list(powerset(defined_subsets)):
    # for size in {10, 20, 35, 50, 70}:
    #     for var in defined_variants:
    #         for special in (True, False):
    #             for profane in (True, False):
    #                 lex = refine(
    #                     build(
    #                         {"american", "english"},
    #                         {"words", "contractions", "upper", "proper-names"},
    #                         size,
    #                         var,
    #                         True,
    #                         True,
    #                     ).union(abbrevs)
    #                 )
    #                 max_performance = max(
    #                     max_performance,
    #                     condition_normalisation.correlation(
    #                         *condition_normalisation.contingency(
    #                             full_raw,
    #                             full_norm,
    #                             lambda x: x in lex,
    #                             pair=True,
    #                         )
    #                     ),
    #                 )
    #                 print(max_performance)
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    normalisations = norm_dict.construct(raw, norm)
    lex = build(
        {"english", "american"},
        {"contractions", "proper-names", "upper", "words"},
        50,
        1,
    )
    lex = refine(lex.union(build_abbreviations()))
    with open(os.path.join(DATA_PATH, "processed/lexicon.txt"), "wb") as f:
        pickle.dump(lex, f)
    evaluate(normalisations, lex)
