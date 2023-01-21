import os
from lexnorm.definitions import LEX_PATH
from lexnorm.evaluation import condition_normalisation
from lexnorm.data import normEval
from lexnorm.definitions import DATA_PATH

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
    lex_path: str = LEX_PATH,
    verbose: bool = False,
) -> set[str]:
    """
    Create word set from SCOWL word lists extending perl script 'mk-list'
    @param languages: including english, american, australian, canadian, british, british_z.
    english should always be included.
    @param subsets: within each language. Including abbreviations, contractions, proper-names, upper, words
    @param max_size: up to 95. 70 is the recommended large size, 50 the medium and 35 the small
    @param max_variant: word variant inclusion level, up to 3 (variants up to level 2 are locale-specific)
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
    if verbose:
        print(f"Component files: {components}")
    for component in components:
        with open(os.path.join(LEX_PATH, component), encoding="iso-8859-1") as file:
            for line in file:
                lexicon.add(line.strip().lower())
    return lexicon


def refine(lexicon: set[str]) -> set[str]:
    # TODO: explore refinements as described in onenote and 1.0 notebook e.g. remove rare single letter chars, add interjections
    # TODO: write test
    filtered_lexicon = set()
    for word in lexicon:
        if len(word) > 1 or word in ["a", "i"]:
            filtered_lexicon.add(word)
    return filtered_lexicon


def evaluate(raw_list, norm_list, lexicon):
    """
    Outputs statistics for a given lexicon
    """
    # TODO: write test, explore evaluation for different lexicons
    a, b, c, d = condition_normalisation.contingency(
        raw_list, norm_list, lambda x: x in lexicon, pair=True
    )
    print(
        f"Correlation of lexicon with normalisation: {condition_normalisation.correlation(a, b, c, d):.2f}"
    )
    print(
        f"Most common un-normalised raw alphanumeric tokens in lexicon: {a.most_common(20)}"
    )
    print(
        f"Most common normalised raw alphanumeric tokens in lexicon: {b.most_common(20)}"
    )
    print(
        f"Most common un-normalised raw alphanumeric tokens not in lexicon: {c.most_common(20)}"
    )
    print(
        f"Most common normalised raw alphanumeric tokens not in lexicon: {d.most_common(20)}"
    )


if __name__ == "__main__":
    lex = build(
        {"english", "american"},
        {"abbreviations", "contractions", "proper-names", "upper", "words"},
        70,
        0,
    )
    full_raw, full_norm = normEval.loadNormData(
        os.path.join(DATA_PATH, "interim/train.txt")
    )
    evaluate(full_raw, full_norm, lex)
