import os
import pickle

import pytest
from spylls.hunspell import Dictionary
from spylls.hunspell.algo.ngram_suggest import precise_affix_score

from lexnorm.data import word2vec, normEval, norm_dict
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.normalise import load_candidates


def test_process_data_file():
    # REPLACE FIRST THREE LINES AS APPROPRIATE
    raw_input, _ = normEval.loadNormData(os.path.join(DATA_PATH, "raw/train.norm"))
    raw_data, norm_data = normEval.loadNormData(
        os.path.join(DATA_PATH, "raw/train.norm")
    )
    cands = load_candidates(os.path.join(DATA_PATH, "hpc/dev_final.cands")).fillna(0)
    # w2v = word2vec.get_vectors(raw_input + raw_data)
    normalisations = norm_dict.construct(raw_data, norm_data)
    with open(os.path.join(DATA_PATH, "processed/task_lexicon.pickle"), "rb") as lf:
        task_lex = pickle.load(lf)
    with open(os.path.join(DATA_PATH, "processed/feature_lexicon.pickle"), "rb") as lf:
        feature_lex = pickle.load(lf)
    spellcheck_dict = Dictionary.from_zip(
        os.path.join(DATA_PATH, "external/hunspell_en_US.zip")
    )
    for i in range(50):
        samp = cands.sample()
        features = samp.to_dict(orient="index")
        cand = list(features.keys())[0]
        raw = features[cand]["raw"]
        # print(features)
        assert all([c in task_lex for c in cand.split()]) if raw != cand else True
        assert is_eligible(cand)
        # assert features[cand]["cosine_to_orig"] == pytest.approx(
        #     w2v.similarity(cand, raw) if (cand in w2v and raw in w2v) else 0
        # )
        assert features[cand]["frac_norms_seen"] == pytest.approx(
            normalisations[raw].get(cand, 0)
            / sum([v for k, v in normalisations[raw].items()])
            if raw in normalisations
            else 0
        )
        if len(raw) > 2:
            assert features[cand]["from_clipping"] == (
                cand.startswith(raw) and raw != cand
            )
        # assert features[cand]["from_embeddings"] == (
        #     (cand in [k for k, v in w2v.most_similar(raw)]) if raw in w2v else 0
        # )
        assert features[cand]["from_original_token"] == (raw == cand)
        assert features[cand]["from_spellcheck"] == (
            raw != cand and cand in spellcheck_dict.suggest(raw)
        )
        if len(raw) > 3:
            assert features[cand]["from_split"] == (
                cand != raw and "".join(cand.split()) == raw
            )
        assert features[cand]["norms_seen"] == (
            normalisations[raw].get(cand, 0) if raw in normalisations else 0
        )
        assert features[cand]["spellcheck_score"] == pytest.approx(
            precise_affix_score(cand, raw, -10, base=0, has_phonetic=False)
        )
        assert features[cand]["length"] == len(cand)
        assert features[cand]["frac_length"] == pytest.approx(len(cand) / len(raw))
        assert features[cand]["norms_seen_orig"] == (
            normalisations[raw].get(raw, 0) if raw in normalisations else 0
        )
        assert features[cand]["frac_norms_seen_orig"] == pytest.approx(
            normalisations[raw].get(raw, 0)
            / sum([v for k, v in normalisations[raw].items()])
            if raw in normalisations
            else 0
        )
        assert features[cand]["length_orig"] == len(raw)
        assert features[cand]["in_feature_lex_orig"] == (raw in feature_lex)
