from lexnorm.data.normEval import loadNormData
import pandas as pd
import os
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.process import create_index
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.random_forest import predict_probs
from collections import Counter
from joblib import load
import statistics


def not_generated(raw, gold, candidates):
    """
    Analyse instances where correct candidate not generated

    :param raw: Raw tweets
    :param gold: Gold tweets
    :param candidates: Candidates dataframe
    :return: Dictionary of {tok_id: (raw token, gold token)} for tok_ids where raw candidate was not generated, and counter
    of (raw token, gold token) tuples
    """
    candidates = create_index(candidates)
    not_generated = {}
    id = -1
    for raw_sent, gold_sent in zip(raw, gold):
        for raw_tok, gold_tok in zip(raw_sent, gold_sent):
            if is_eligible(raw_tok):
                id += 1
                if not gold_tok in candidates.loc[candidates["tok_id"] == id].index:
                    not_generated[id] = (raw_tok, gold_tok)
    # recall of all candidates
    recall_all = ((id + 1) - len(not_generated)) / (id + 1)
    print(f"Recall of all candidates: {recall_all*100:.2f}")
    print(
        f"Most common ungenerated normalisations: {Counter(not_generated.values()).most_common(10)}"
    )
    return not_generated


def generated_rank(raw, gold, candidates_with_preds, verbose=True, top_n=2):
    """
    Analyse instances when correct candidate generated

    :param raw: Raw tweets
    :param gold: Gold tweets
    :param candidates_with_preds: Dataframe of generated candidates with predictions from model (from predict_probs)
    :param verbose: if to print analysis or just return statistics
    :param top_n: top n predictions to evaluate recall on
    """
    candidates = create_index(candidates_with_preds).sort_values(
        "probs", ascending=False
    )
    candidates["candidate"] = candidates.index.values
    not_top = {}
    ranks = {}
    correct_probs = []
    correct_top_probs = []
    incorrect_probs_generated = []
    incorrect_probs_not_generated = []
    id = -1
    for raw_sent, gold_sent in zip(raw, gold):
        for raw_tok, gold_tok in zip(raw_sent, gold_sent):
            if is_eligible(raw_tok):
                id += 1
                cur_cands = candidates.loc[candidates["tok_id"] == id].reset_index()
                if gold_tok in set(cur_cands["candidate"]):
                    ranks[id] = cur_cands.index[
                        cur_cands["candidate"] == gold_tok
                    ].tolist()[0]
                    correct_probs.append(cur_cands.iloc[ranks[id]]["probs"])
                    if not ranks[id]:
                        correct_top_probs.append(cur_cands.iloc[ranks[id]]["probs"])
                    if ranks[id]:
                        not_top[id] = (
                            raw_tok,
                            cur_cands.iloc[0]["candidate"],
                            gold_tok,
                            ranks[id],
                            cur_cands.iloc[0]["probs"]
                            - cur_cands.iloc[ranks[id]]["probs"],
                        )
                if gold_tok not in set(cur_cands["candidate"]):
                    incorrect_probs_not_generated.append(cur_cands.iloc[0]["probs"])
                elif ranks[id]:
                    incorrect_probs_generated.append(cur_cands.iloc[0]["probs"])
    median_rank = statistics.median(sorted(ranks.values()))
    mean_rank = statistics.mean(ranks.values())
    mean_correct_prob = statistics.mean(correct_probs)
    mean_incorrect_top_generated_prob = statistics.mean(incorrect_probs_generated)
    mean_incorrect_top_not_generated_prob = statistics.mean(
        incorrect_probs_not_generated
    )
    mean_correct_top_prob = statistics.mean(correct_top_probs)
    median_rank_not_top = statistics.median(sorted([v[3] for v in not_top.values()]))
    mean_rank_not_top = statistics.mean([v[3] for v in not_top.values()])
    mean_prob_diff_not_top = statistics.mean([v[4] for v in not_top.values()])
    percentage_generated_top = len([v for v in ranks.values() if not v]) / len(ranks)
    top_n_recall = len([v for v in ranks.values() if v < top_n]) / (id + 1)
    if verbose:
        print(
            f"Median rank of correct candidates: {median_rank}, "
            f"mean: {mean_rank:.2f}"
        )
        print(f"Mean probability of correct candidates: {mean_correct_prob:.3f}")
        print(
            f"Mean probability of correct candidates ranked first: {mean_correct_top_prob:.3f}"
        )
        print(
            f"Mean probability of incorrect candidates ranked first when correct candidate generated: {mean_incorrect_top_generated_prob:.3f}"
        )
        print(
            f"Mean probability of incorrect candidates ranked first when correct candidate not generated: {mean_incorrect_top_not_generated_prob:.3f}"
        )
        print(
            f"Median rank of correct candidates not ranked first: {median_rank_not_top}, "
            f"mean: {mean_rank_not_top:.2f}"
        )
        print(
            f"Mean probability difference of correct candidates not ranked first to first ranked candidate: {mean_prob_diff_not_top:.3f}"
        )
        print(
            f"Percentage of correct candidates ranked first: {percentage_generated_top * 100:.2f}"
        )
        print(f"Recall of top {top_n} candidate(s): {top_n_recall * 100:.2f}")
        print(
            f"Most common ranking errors: {Counter([v[:3] for v in not_top.values()]).most_common(10)}"
        )
        print(
            f"Closest ranking errors: {sorted(not_top.values(), key = lambda x: x[4])[:10]}"
        )
    return (
        ranks,
        not_top,
        correct_probs,
        correct_top_probs,
        incorrect_probs_generated,
        incorrect_probs_not_generated,
        top_n_recall,
    )


if __name__ == "__main__":
    data = pd.read_csv(
        os.path.join(DATA_PATH, "hpc/dev_ngrams.txt"),
        index_col=0,
        keep_default_na=False,
        na_values="",
    ).sample(
        frac=1,
        # random_state=42,
    )
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    clf = load(os.path.join(DATA_PATH, "../models/rf.joblib"))
    not_generated(raw, norm, data)
    ranks = generated_rank(
        raw, norm, predict_probs(clf, os.path.join(DATA_PATH, "hpc/dev_ngrams.txt"))
    )
