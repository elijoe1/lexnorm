from lexnorm.data.normEval import loadNormData
import pandas as pd
import os
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.process import create_index, link_to_gold
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.normalise import load_candidates
from lexnorm.models.random_forest import predict_probs
from lexnorm.data.analyse import analyse
from collections import Counter
from joblib import load
import statistics
import numpy as np


def modules(gold_path, candidates, verbose=True):
    """
    Analyse recall of candidate generation modules (split, clipping, spellcheck, norm_dict, embeddings) on normalisations (e.g.
    where the correct normalisation is not the original token), and give average candidates generated/token over whole data.
    Obviously recall will be 100 for original token module and all modules when raw == norm, and 0 for original token
    module for raw != norm - the only remaining recall is for the remaining modules when raw != norm as calculated here.

    :param raw: Raw tweets
    :param gold: Gold tweets
    :param candidates: Candidates dataframe
    :param verbose: If want to print analysis or just return statistics
    :return: A dictionary giving column (i.e. module) : (recall solo, recall without, set of tok_ids uniquely correct for),
    and combined recall of all modules on normalisations
    """
    columns = [
        "from_split",
        "from_clipping",
        "spellcheck_rank",
        "norms_seen",
        "embeddings_rank",
    ]
    raw, norm = loadNormData(gold_path)
    candidates = link_to_gold(candidates, raw, norm)
    num_eligible, num_norms = analyse(gold_path)
    modules = {}
    resp = {}
    for column in columns:
        resp[column] = (
            len(candidates.loc[~np.isnan(candidates[column])].index) / num_eligible
        )
        correct = candidates.loc[
            # can't check for equality with np.nan as this will always return false!
            (~np.isnan(candidates[column]))
            & (np.isnan(candidates.from_original_token))
            & (candidates.correct)
        ][[column, "correct", "tok_id"]]
        modules[column] = set(correct.tok_id.values.tolist())
    for column in columns:
        recall_solo = len(modules[column]) / num_norms
        without = set().union(*[v for k, v in modules.items() if k != column])
        recall_without = len(without) / num_norms
        unique = modules[column] - without
        resp[column] = (recall_solo, recall_without, unique, resp[column])
        if verbose:
            print(column.upper())
            print(f"Solo recall (over normalisations): {recall_solo*100:.2f}")
            print(f"Ablation recall (over normalisations): {recall_without*100:.2f}")
            print(
                f"Unique recall (over normalisations): {len(unique)/num_norms * 100:.2f}"
            )
            print(f"Average candidates generated per token: {resp[column][3]:.2f}")
    combined_recall = len(set().union(*[v for v in modules.values()])) / num_norms
    if verbose:
        print(f"COMBINED RECALL: {combined_recall * 100:.2f}")
        # this will likely be lower than the sum for each module as overlapping suggestions will be present
        print(f"COMIBINED CANDIDATES/TOKEN: {len(candidates.index) / num_eligible:.2f}")
    return resp, combined_recall


def not_generated(raw, gold, candidates):
    """
    Analyse instances where correct candidate not generated (over normalisations only as obviously will always be
    generated if raw_tok == norm_tok

    :param raw: Raw tweets
    :param gold: Gold tweets
    :param candidates: Candidates dataframe
    :return: Dictionary of {tok_id: (raw token, gold token)} for tok_ids where raw candidate was not generated
    """
    not_generated = {}
    id = -1
    norms = 0
    for raw_sent, gold_sent in zip(raw, gold):
        for raw_tok, gold_tok in zip(raw_sent, gold_sent):
            if is_eligible(raw_tok):
                id += 1
                if raw_tok != gold_tok:
                    norms += 1
                    if not gold_tok in candidates.loc[candidates["tok_id"] == id].index:
                        not_generated[id] = (raw_tok, gold_tok)
    # recall of all candidates
    recall_all = (norms - len(not_generated)) / norms
    print(f"Recall of all candidates: {recall_all*100:.2f}")
    print(
        f"Most common ungenerated normalisations: {Counter(not_generated.values()).most_common(10)}"
    )
    return not_generated


def ranking(raw, gold, candidates_with_preds, verbose=True, top_n=2, norms_only=True):
    """
    Analyses model predictions when correct candidate generated

    :param raw: Raw tweets
    :param gold: Gold tweets
    :param candidates_with_preds: Dataframe of generated candidates with predictions from model (from predict_probs)
    :param verbose: if to print analysis or just return statistics
    :param top_n: top n predictions to evaluate recall on
    :param norms_only: if evaluating over normalisations only, or over all data
    """
    candidates = candidates_with_preds.sort_values("probs", ascending=False)
    candidates["candidate"] = candidates.index.values
    not_top = {}
    ranks = {}
    correct_probs = []
    correct_top_probs = []
    incorrect_probs_generated = []
    incorrect_probs_not_generated = []
    id = -1
    norms = 0
    for raw_sent, gold_sent in zip(raw, gold):
        for raw_tok, gold_tok in zip(raw_sent, gold_sent):
            if is_eligible(raw_tok):
                id += 1
                cur_cands = candidates.loc[candidates["tok_id"] == id].reset_index()
                if not norms_only or raw_tok != gold_tok:
                    norms += 1
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
    top_n_recall = len([v for v in ranks.values() if v < top_n]) / (
        id + 1 if not norms_only else norms
    )
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
    data = load_candidates(
        os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"), shuffle=True
    )
    raw, norm = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    clf = load(os.path.join(DATA_PATH, "../models/rf.joblib"))
    ranks = ranking(
        raw, norm, predict_probs(clf, os.path.join(DATA_PATH, "hpc/dev_pipeline.txt"))
    )
    not_generated(raw, norm, data)
    modules(os.path.join(DATA_PATH, "raw/dev.norm"), data)
