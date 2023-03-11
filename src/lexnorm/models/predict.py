from lexnorm.models.normalise import prep_test


def predict_probs(model, candidates):
    features = prep_test(candidates)
    probs = model.predict_proba(features)
    candidates = candidates.copy()
    candidates["probs"] = probs[:, 1]
    return candidates


def predict_normalisations(dataframe, threshold=0.5):
    # 0.5 is not an arbitrary threshold - if above .predict function of classifier would predict class 1
    # as takes class with highest proba and there are only two classes. NOTE that .predict_proba should not be interpreted
    # as confidence level of class for random forest - just number input to decision function.
    # ISSUE: if tie for highest probability, just chooses arbitrarily
    pred_df = dataframe.sort_values("probs", ascending=False).drop_duplicates(
        ["tok_id"]
    )
    pred_df = pred_df.loc[pred_df.probs >= threshold]
    # pred_df = pred_df.sort_values(["tok_id"]).index.tolist()
    pred_df["candidate"] = pred_df.index.values
    # as values are singleton lists
    return {
        k: v[0]
        for k, v in pred_df.groupby("tok_id")["candidate"].apply(list).to_dict().items()
    }
