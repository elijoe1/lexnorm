from collections import Counter

from lexnorm.data.normEval import err
from lexnorm.generate_extract.filtering import is_eligible


# ADAPTED FROM CODE IN TASK REPOSITORY (normEval.py)
def evaluate_predictions(raw, gold, pred):
    cor = 0
    changed = 0
    total = 0
    errors = {}

    if len(gold) != len(pred):
        err(
            "Error: gold normalization contains a different number of sentences("
            + str(len(gold))
            + ") compared to system output("
            + str(len(pred))
            + ")"
        )

    id = -1
    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            err(
                "Error: a sentence has a different length in your output, check the order of the sentences"
            )
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            if is_eligible(wordRaw):
                id += 1
            wordRaw = wordRaw.lower()
            wordGold = wordGold.lower()
            wordPred = wordPred.lower()
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            else:
                if is_eligible(wordRaw):
                    errors[id] = (wordRaw, wordPred, wordGold)
            total += 1

    accuracy = cor / total
    lai = (total - changed) / total
    error = (accuracy - lai) / (1 - lai)
    precision, recall, f1 = precision_recall_f1(raw, gold, pred)

    print("Baseline acc.(LAI): {:.2f}".format(lai * 100))
    print("Accuracy:           {:.2f}".format(accuracy * 100))
    print("ERR:                {:.2f}".format(error * 100))
    print(
        f"Most common wrongly un-normalised: {Counter([v for v in errors.values() if v[0] == v[1]]).most_common(10)}"
    )
    print(
        f"Most common incorrect normalisation choice: {Counter([v for v in errors.values() if v[0] != v[2] and v[0] != v[1]]).most_common(10)}"
    )
    print(
        f"Most common false positives: {Counter([v for v in errors.values() if v[0] == v[2]]).most_common(10)}"
    )

    return lai, accuracy, error, precision, recall, f1, errors


def precision_recall_f1(raw, gold, pred):
    fp = 0
    fn = 0
    tp = 0

    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            err(
                "Error: a sentence has a different length in your output, check the order of the sentences"
            )
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            wordRaw = wordRaw.lower()
            wordGold = wordGold.lower()
            wordPred = wordPred.lower()
            if wordRaw != wordGold and wordPred != wordGold:
                # annotators normalised, system normalised but incorrectly / did not normalise
                fn += 1
            elif wordRaw == wordGold and wordPred != wordRaw:
                # annotators did not normalise, system normalised
                fp += 1
            elif wordRaw != wordGold and wordPred == wordGold:
                # annotators normalised, system normalised correctly
                tp += 1
            # other category is TN (both did not normalise) but this isn't needed for F1

    # precision doesn't take into account normalising wrongly - this is done in recall only to avoid double counting
    # which is present in WNUT 2015, penalising correct decisions to normalise over incorrect decisions not to.
    # precision and recall supplement ERR as tell us balance between eagerness and conservativeness of model
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("Precision: {:.2f}".format(precision * 100))
    print("Recall: {:.2f}".format(recall * 100))
    print("F1: {:.2f}".format(f1 * 100))

    return precision, recall, f1
