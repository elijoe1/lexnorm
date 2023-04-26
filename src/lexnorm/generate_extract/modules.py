import pandas as pd

from spylls.hunspell.algo.ngram_suggest import precise_affix_score

from lexnorm.generate_extract.filtering import is_eligible


def original_token(tok):
    # From Monoise
    # Required if explicit detect step is skipped, as all tokens must be replaced by one of the candidates.
    candidate = pd.DataFrame(columns=["from_original_token"])
    candidate.loc[tok] = {"from_original_token": 1}
    return candidate


def spellcheck(tok, dictionary):
    # From Monoise
    candidates = pd.DataFrame(columns=["from_spellcheck", "spellcheck_score"])
    for c in dictionary.suggest(tok):
        # Checks if c.islower() before using - capitalised candidates cause issues with merging dataframe rows
        # Check for ineligible suggestions as these are never the correct normalisation
        if is_eligible(c) and c.islower() and c != tok:
            candidates.loc[c] = {
                "from_spellcheck": 1,
                "spellcheck_score": precise_affix_score(
                    c, tok, -10, base=0, has_phonetic=False
                ),
            }
    return candidates


def split(tok, lex):
    # From Monoise
    # Hypothesise split at every position and check if both words are in lexicon.
    candidates = pd.DataFrame(columns=["from_split"])
    if len(tok) <= 3:
        return candidates
    for pos in range(1, len(tok)):
        left = tok[:pos]
        right = tok[pos:]
        if left in lex and right in lex:
            candidates.loc[" ".join([left, right])] = {"from_split": 1}
    return candidates


def clipping(tok, lex):
    # From Monoise
    # Gives all words in lexicon that have tok as a prefix.
    candidates = pd.DataFrame(columns=["from_clipping"])
    if len(tok) <= 2:
        return candidates
    for c in lex:
        if c.startswith(tok) and c != tok:
            candidates.loc[c] = {"from_clipping": 1}
    return candidates


def norm_lookup(tok, dict):
    # From Monoise
    # Gives everything raw token seen to normalise to.
    # Don't need from_lookup as this is collinear with norms_seen
    candidates = pd.DataFrame(columns=["norms_seen", "frac_norms_seen"])
    normalisations = dict.get(tok, {}).items()
    total = sum([v for k, v in normalisations])
    for k, v in normalisations:
        candidates.loc[k] = {"norms_seen": v, "frac_norms_seen": v / total}
    return candidates


def word_embeddings(tok, vectors, lex, threshold=0):
    # From Monoise
    # Issue: lower cased query means embeddings only found for lowercase word!
    # Issue: antonyms also often present in same contexts.
    # Using Twitter embeddings from van der Goot - based on distributional hypothesis to find tokens with similar semantics.
    candidates = pd.DataFrame(columns=["from_embeddings", "cosine_to_orig"])
    cand_dict = {}
    if tok in vectors:
        for c in vectors.similar_by_vector(tok, topn=10):
            # This check is needed as can produce ineligible suggestions
            # Trying lowercase only again:
            if (
                is_eligible(c[0])
                and c[1] >= threshold
                and c[0].islower()
                and c[0] in lex
            ):
                cand_dict[c[0]] = {
                    "from_embeddings": 1,
                    "cosine_to_orig": c[1],
                }
    for cand, features in cand_dict.items():
        candidates.loc[cand] = features
    return candidates
