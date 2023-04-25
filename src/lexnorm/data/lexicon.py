import os
import pickle
from itertools import chain, combinations

import requests

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
    remove_offensive: bool = False,
    lex_path: str = LEX_PATH,
    verbose: bool = False,
) -> set[str]:
    """
    Create (lowercase) word set from SCOWL word lists extending perl script 'mk-list'

    :param languages: including english, american, australian, canadian, british, british_z.
    english should always be included.
    :param subsets: within each language. Including abbreviations, contractions, proper-names, upper, words
    :param max_size: up to 95. 70 is the recommended large size, 50 the medium and 35 the small
    :param max_variant: word variant inclusion level, up to 3 (variants up to level 2 are locale-specific)
    :param remove_offensive: if offensive terms should be removed (from misc folder in scowl)
    :param special: if the special word lists (hacker and roman numerals) should be included if the size is large enough
    :param lex_path: path to SCOWL word lists
    :param verbose: if the word lists used should be output
    :return: set containing all words in corresponding word lists
    """
    lexicon = set()
    offensive = set()
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
    if remove_offensive:
        for component in ["../misc/offensive.1", "../misc/offensive.2"]:
            with open(os.path.join(LEX_PATH, component), encoding="iso-8859-1") as file:
                for line in file:
                    offensive.add(line.strip().lower())
    if verbose:
        print(f"Component files: {components}")
    for component in components:
        with open(os.path.join(LEX_PATH, component), encoding="iso-8859-1") as file:
            for line in file:
                lexicon.add(line.strip().lower())
    return lexicon - offensive


def refine(lexicon: set[str]) -> set[str]:
    filtered_lexicon = set()
    for word in lexicon:
        if len(word) == 1 and word not in ["a", "i"]:
            continue
        if len(
            word
        ) == 2 and word not in "am, an, as, at, ax, be, by, do, go, he, if, in, is, it, me, my, no, of, on, or, ox, so, to, up, us, we".split(
            ", "
        ):
            continue
        filtered_lexicon.add(word)
    return filtered_lexicon


def build_interjections():
    abbrev_lex = set()
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:English_internet_laughter_slang",
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


def evaluate(raw, norm, lexicon, verbose=False):
    """
    Outputs statistics for a given lexicon
    """
    a, b, c, d = condition_normalisation.contingency(
        raw, norm, lambda x: x[0] in lexicon
    )
    correlation = condition_normalisation.correlation(a, b, c, d)
    if verbose:
        print(f"Correlation of in lexicon with normalisation: {correlation:.3f}")
        print(f"Most common normalised raw tokens in lexicon: {b.most_common(20)}")
        print(
            f"Most common un-normalised raw tokens not in lexicon: {c.most_common(20)}"
        )
        a, b, c, d = condition_normalisation.contingency(
            raw,
            norm,
            lambda x: any(word not in lexicon for word in x[1].split()),
        )
        print(
            f"Most common token normalisations not in lexicon: {b.most_common(20)}: {sum(b.values())/sum((b+d).values())*100:.2f}% of normalisations."
        )
    return correlation


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_best_lexicon():
    raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    abbrevs = build_interjections()
    max_performance = 0
    best_params = []
    for locale in list(defined_languages - {"english"}):
        for subsets in list(powerset(defined_subsets)):
            for size in defined_sizes:
                for var in defined_variants:
                    for special in (True, False):
                        for remove_offensive in (True, False):
                            for do_refine in (True, False):
                                for add_intj in (True, False):
                                    lex = build(
                                        {locale, "english"},
                                        set(subsets),
                                        size,
                                        var,
                                        special,
                                        remove_offensive,
                                    )
                                    if add_intj:
                                        lex = lex.union(abbrevs)
                                    if do_refine:
                                        lex = refine(lex)
                                    corr = evaluate(raw, norm, lex)
                                    print(
                                        f"{locale}, {subsets}, {size}, {var}, {special}, {remove_offensive}, {do_refine}, {add_intj}: {corr:.3f}"
                                    )
                                    if corr > max_performance:
                                        max_performance = corr
                                        best_params = [
                                            locale,
                                            subsets,
                                            size,
                                            var,
                                            special,
                                            remove_offensive,
                                            do_refine,
                                            add_intj,
                                        ]
    return best_params


if __name__ == "__main__":
    # find_best_lexicon()
    lex = build(
        {"english", "american"},
        {"contractions", "proper-names", "upper", "words", "abbreviations"},
        70,
        1,
        True,
        False,
    )
    with open(os.path.join(DATA_PATH, "processed/task_lexicon.txt"), "wb") as f:
        pickle.dump(lex, f)
    lex = build(
        {"english", "american"},
        {"contractions", "proper-names", "upper", "words"},
        50,
        1,
        True,
        True,
    )
    lex = refine(lex.union(build_interjections()))
    with open(os.path.join(DATA_PATH, "processed/feature_lexicon.txt"), "wb") as f:
        pickle.dump(lex, f)
    # raw, norm = normEval.loadNormData(os.path.join(DATA_PATH, "processed/combined.txt"))
    # evaluate(raw, norm, lex, True)
