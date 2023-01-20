import os
from lexnorm.definitions import LEX_PATH

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


def build_lexicon(
    languages: set[str],
    subsets: set[str],
    max_size: int = 70,
    max_variant: int = 0,
    special: bool = True,
    lex_path: str = LEX_PATH,
    verbose: bool = False,
) -> set[str]:
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


def filter_lexicon(lexicon: set[str]) -> set[str]:
    filtered_lexicon = set()
    for word in lexicon:
        if len(word) > 1 or word in ["a", "i"]:
            filtered_lexicon.add(word)
    return filtered_lexicon


if __name__ == "__main__":
    build_lexicon(
        {"english", "british"},
        {"abbreviations", "contractions", "proper-names", "upper", "words"},
        35,
        2,
    )
