from sklearn.model_selection import KFold

from lexnorm.data.normEval import loadNormData
import pandas as pd
import os
from lexnorm.definitions import DATA_PATH
from lexnorm.generate_extract.candidates import create_index, link_to_gold
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.classifiers import train_predict_evaluate_cv
from lexnorm.models.normalise import load_candidates
from lexnorm.models.predict import predict_probs
from lexnorm.evaluation.analyse import analyse, get_tokens_from_ids
from collections import Counter
from joblib import load
import statistics
import numpy as np


def modules(raw, norm, candidates, verbose=True):
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
        "from_spellcheck",
        "norms_seen",
        "from_embeddings",
    ]
    candidates = link_to_gold(candidates, raw, norm)
    num_eligible, num_norms = analyse(raw, norm)
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
        unique = get_tokens_from_ids(modules[column] - without, raw, norm)
        resp[column] = (
            recall_solo,
            recall_without,
            unique,
            resp[column],
        )
        if verbose:
            print(column.upper())
            print(f"Solo recall (over normalisations): {recall_solo*100:.2f}")
            print(f"Ablation recall (over normalisations): {recall_without*100:.2f}")
            print(
                f"Unique recall (over normalisations): {len(unique)/num_norms * 100:.2f}"
            )
            print(f"Unique normalisations found: {unique}")
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


def ranking(raw, gold, candidates_with_preds, verbose=True, top_n=1, norms_only=True):
    """
    Analyses model predictions when correct candidate generated

    ISSUE: for tied probabilities, ordering of candidates arbitrary (depends on shuffle). If we do not have joint ranks,
    this would lead to very variable rank-related metrics.
    The biggest issue is that a lot of probabilities are 0, so if the correct candidate is in that set, can be huge range
    of positions it can be in. Generally few candidates are ranked first, so the rank metrics for top rankings specifically
    do not change very much with shuffling or with joint ranks (what we care about the most).

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
    top_n_count = []
    id = -1
    norms = 0
    for raw_sent, gold_sent in zip(raw, gold):
        for raw_tok, gold_tok in zip(raw_sent, gold_sent):
            if is_eligible(raw_tok):
                id += 1
                cur_cands = candidates.loc[candidates["tok_id"] == id].reset_index()
                probs = cur_cands["probs"].tolist()
                top_n_count.append(
                    sum(
                        i[1]
                        for i in sorted(
                            Counter(probs).items(), key=lambda x: x[0], reverse=True
                        )[:top_n]
                    )
                )
                probs = sorted(list(set(probs)), reverse=True)
                if not norms_only or raw_tok != gold_tok:
                    norms += 1
                    if gold_tok in set(cur_cands["candidate"]):
                        ranks[id] = probs.index(
                            cur_cands.loc[cur_cands["candidate"] == gold_tok][
                                "probs"
                            ].tolist()[0]
                        )
                        correct_probs.append(probs[ranks[id]])
                        if not ranks[id]:
                            correct_top_probs.append(probs[ranks[id]])
                        if ranks[id]:
                            not_top[id] = (
                                raw_tok,
                                gold_tok,
                                ranks[id],
                                probs[0] - probs[ranks[id]],
                            )
                    if gold_tok not in set(cur_cands["candidate"]):
                        incorrect_probs_not_generated.append(probs[0])
                    elif ranks[id]:
                        incorrect_probs_generated.append(probs[0])
    median_rank = statistics.median(sorted(ranks.values()))
    mean_rank = statistics.mean(ranks.values())
    mean_correct_prob = statistics.mean(correct_probs)
    mean_incorrect_top_generated_prob = statistics.mean(incorrect_probs_generated)
    mean_incorrect_top_not_generated_prob = statistics.mean(
        incorrect_probs_not_generated
    )
    mean_correct_top_prob = statistics.mean(correct_top_probs)
    median_rank_not_top = statistics.median(sorted([v[2] for v in not_top.values()]))
    mean_rank_not_top = statistics.mean([v[2] for v in not_top.values()])
    mean_prob_diff_not_top = statistics.mean([v[3] for v in not_top.values()])
    percentage_generated_top = len([v for v in ranks.values() if not v]) / len(ranks)
    top_n_recall = len([v for v in ranks.values() if v < top_n]) / (
        id + 1 if not norms_only else norms
    )
    if verbose:
        print(
            f"Median rank of correct candidates: {median_rank}, "
            f"mean: {mean_rank:.2f}"
        )
        print(
            f"Median rank of correct candidates not ranked first: {median_rank_not_top}, "
            f"mean: {mean_rank_not_top:.2f}"
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
            f"Mean probability difference of correct candidates not ranked first to first ranked candidate: {mean_prob_diff_not_top:.3f}"
        )
        print(
            f"Percentage of correct candidates ranked first: {percentage_generated_top * 100:.2f}"
        )
        print(f"Recall of top {top_n} ranked candidate(s): {top_n_recall * 100:.2f}")
        print(
            f"Mean number of candidates ranked top {top_n}: {statistics.mean(top_n_count):.2f}, median: {statistics.median(sorted(top_n_count))}"
        )
        print(
            f"Most common ranking errors: {Counter([v[:2] for v in not_top.values()]).most_common(10)}"
        )
        print(
            f"Closest ranking errors: {sorted(not_top.values(), key = lambda x: x[3])[:10]}"
        )
    return (
        ranks,
        not_top,
        correct_probs,
        correct_top_probs,
        incorrect_probs_generated,
        incorrect_probs_not_generated,
        top_n_recall,
        top_n_count,
    )


def feature_ablation(model, output_path):
    features = [
        "cosine_to_orig",
        "frac_norms_seen",
        "from_clipping",
        "from_embeddings",
        "from_original_token",
        "from_spellcheck",
        "from_split",
        "norms_seen",
        "spellcheck_score",
        "length",
        "frac_length",
        "same_order",
        "in_feature_lex_orig",
        "wiki_uni_cand",
        "twitter_uni_cand",
        "wiki_bi_prev_cand",
        "wiki_bi_cand_next",
        "twitter_bi_prev_cand",
        "twitter_bi_cand_next",
        "twitter_uni_cand_orig",
        "twitter_bi_prev_cand_orig",
        "twitter_bi_cand_next_orig",
        "wiki_uni_cand_orig",
        "wiki_bi_prev_cand_orig",
        "wiki_bi_cand_next_orig",
        "length_orig",
        "norms_seen_orig",
        "frac_norms_seen_orig",
    ]
    scores = {}
    for feature in features:
        # TODO do on combined and test
        scores[feature] = train_predict_evaluate_cv(
            model,
            None,
            os.path.join(DATA_PATH, "processed/combined.txt"),
            os.path.join(DATA_PATH, "hpc/cv"),
            None,
            train_first=True,
            drop_features=feature,
        )
    with open(output_path, "w") as f:
        f.write(str(scores))


def evaluate_cv(model_dir, tweets_path, df_dir):
    raw, norm = loadNormData(tweets_path)
    raw = np.array(raw, dtype=object)
    norm = np.array(norm, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    comb_raw = []
    comb_norm = []
    comb_pred_df = pd.DataFrame()
    offset = 0
    for i, folds in enumerate(kf.split(raw, norm)):
        test_df = create_index(
            load_candidates(
                os.path.join(DATA_PATH, df_dir, f"test_{i}.cands"), shuffle=True
            ),
            offset,
        )
        offset += test_df.tok_id.nunique()
        train_idx, test_idx = folds
        comb_raw += raw[test_idx].tolist()
        comb_norm += norm[test_idx].tolist()
        clf = load(os.path.join(DATA_PATH, model_dir, f"{i}.joblib"))
        comb_pred_df = pd.concat([comb_pred_df, predict_probs(clf, test_df)])
    ranking(comb_raw, comb_norm, comb_pred_df)
    not_generated(comb_raw, comb_norm, comb_pred_df)
    modules(comb_raw, comb_norm, comb_pred_df)


def evaluate(model_path, tweets_path, df_path):
    # NOTE assumes create_index already saved to file, as is fine with non-cv operation
    raw, norm = loadNormData(tweets_path)
    test_df = load_candidates(df_path, shuffle=True)
    clf = load(model_path)
    ranking(raw, norm, predict_probs(clf, test_df))
    not_generated(raw, norm, test_df)
    modules(raw, norm, test_df)


if __name__ == "__main__":
    evaluate(
        os.path.join(DATA_PATH, "../models/rf.joblib"),
        os.path.join(DATA_PATH, "raw/test.norm"),
        os.path.join(DATA_PATH, "hpc/test.cands"),
    )
    # evaluate_cv(
    #     os.path.join(DATA_PATH, "../models/rf"),
    #     os.path.join(DATA_PATH, "processed/combined.txt"),
    #     os.path.join(DATA_PATH, "hpc/cv"),
    # )
