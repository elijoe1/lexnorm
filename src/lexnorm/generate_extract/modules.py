import pandas as pd

from lexnorm.generate_extract.filtering import is_eligible


def original_token(tok):
    # From Monoise
    # Required if explicit detect step is skipped, as all tokens must be replaced by one of the candidates.
    candidate = pd.DataFrame(columns=["from_original_token"])
    candidate.loc[tok] = {"from_original_token": 1}
    return candidate


def spellcheck(tok, dictionary):
    # TODO: essentially no control over details e.g. limits to different types of suggestions.
    #   in addition, loading in custom lexicon basically infeasible as bizarre format required
    # From Monoise
    # Don't need from_spellcheck feature as this is collinear with spellcheck_rank
    candidates = pd.DataFrame(columns=["spellcheck_rank"])
    # So we can use 0 to fill NaNs, as otherwise the top ranked word would be indistinguishable - big issue!
    rank = 1
    for c in dictionary.suggest(tok):
        # Previously, checked if c.islower() before using.
        # I think it makes more sense to generate all suggestions and make them lowercase.
        # Obviously we have lost some information from the input being lowercase, but we can't do anything about that.
        candidates.loc[c.lower()] = {"spellcheck_rank": rank}
        rank += 1
    return candidates


def split(tok, lex):
    # From Monoise
    # Hypothesise split at every position and check if both words are in lexicon.
    # TODO: currently doesn't hypothesise for lengths <= 3 - explore? Allow more than one split?
    candidates = pd.DataFrame(columns=["from_split"])
    if len(tok) < 3:
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
    # TODO: currently only considers for length >= 2. Can produce huge amount of candidates - increase threshold? Prune?
    # TODO: reverse direction? To capture repetition of last character for emphasis and so on.
    candidates = pd.DataFrame(columns=["from_clipping"])
    if len(tok) < 2:
        return candidates
    for c in lex:
        if c.startswith(tok) and c != tok:
            candidates.loc[c] = {"from_clipping": 1}
    return candidates


def norm_lookup(tok, normalisations):
    # TODO: make use of external normalisation dictionaries?
    # From Monoise
    # Gives everything raw token seen to normalise to.
    # Don't need from_lookup as this is collinear with norms_seen
    candidates = pd.DataFrame(columns=["norms_seen"])
    for k, v in normalisations.get(tok, {}).items():
        candidates.loc[k] = {"norms_seen": v}
    return candidates


def word_embeddings(tok, vectors, threshold=0):
    # TODO could reimplement word2vec with keras. Experiment if replacing with new embeddings is worth the extra cost.
    #  Could even create twitter embeddings from scratch? Would clean data as VDG did before creating embeddings.
    #  Cosine similarity threshold? Perhaps until below threshold or after 10 candidates, whatever is later.
    # From Monoise
    # Issue: lower cased query means embeddings only found for lowercase word!
    # Issue: antonyms also often present in same contexts.
    # Using Twitter embeddings from van der Goot - based on distributional hypothesis to find tokens with similar semantics.
    # Don't need from_word_embeddings as collinear with embeddings_rank
    candidates = pd.DataFrame(columns=["cosine_to_orig", "embeddings_rank"])
    cand_dict = {}
    if tok in vectors:
        # Previously checked if c was lower before returning. But I think it's better to generate all and make lowercase.
        cand_list = [
            (c[0].lower(), c[1])
            for c in vectors.similar_by_vector(tok, topn=10)
            # This is the only module that can produce ineligible suggestions, so this check is needed.
            if is_eligible(c[0]) and c[1] >= threshold
        ]
        # Get the highest cosine similarity for each lowercase suggestion (needed as several suggestions may lowercase
        # to the same thing.
        for cand in cand_list:
            if cand[1] > cand_dict.get(cand[0], 0):
                cand_dict[cand[0]] = cand[1]
    for rank, c in enumerate(
        sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)
    ):
        k, v = c
        candidates.loc[k] = {
            "cosine_to_orig": v,
            # So that 0 can be used to fill NaN values. This may be unnecessary as cosine similarity is already being used as a feature.
            "embeddings_rank": rank + 1,
        }
    return candidates
