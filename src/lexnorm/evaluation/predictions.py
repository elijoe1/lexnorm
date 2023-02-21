from collections import Counter

from lexnorm.data.normEval import err

# ADAPTED FROM CODE IN TASK REPOSITORY (normEval.py)
def evaluate_predictions(raw, gold, pred):
    cor = 0
    changed = 0
    total = 0
    errors = Counter()

    if len(gold) != len(pred):
        err(
            "Error: gold normalization contains a different number of sentences("
            + str(len(gold))
            + ") compared to system output("
            + str(len(pred))
            + ")"
        )

    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            err(
                "Error: a sentence has a different length in your output, check the order of the sentences"
            )
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            wordRaw = wordRaw.lower()
            wordGold = wordGold.lower()
            wordPred = wordPred.lower()
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            else:
                # TODO give tweet and token index of errors for comparison
                errors.update([(wordRaw, wordGold, wordPred)])
            total += 1

    accuracy = cor / total
    lai = (total - changed) / total
    error = (accuracy - lai) / (1 - lai)

    print("Baseline acc.(LAI): {:.2f}".format(lai * 100))
    print("Accuracy:           {:.2f}".format(accuracy * 100))
    print("ERR:                {:.2f}".format(error * 100))
    print(errors.most_common())

    return lai, accuracy, error, errors


# TODO: for two classifiers, compare predictions
