from lexnorm.data.baseline import write
from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.models.prepare_df import prep_test


def predict_probs(model, candidates):
    features = prep_test(candidates)
    probs = model.predict_proba(features)
    candidates = candidates.copy()
    candidates["probs"] = probs[:, 1]
    return candidates


def predict_candidates(dataframe, threshold=0.5):
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


def predict_normalisation(raw, pred_toks: dict, output_path=None, baseline_preds=None):
    tok_id = -1
    pred_tweets = []
    for j, tweet in enumerate(raw):
        pred_tweet = []
        for i, tok in enumerate(tweet):
            if is_eligible(tok):
                tok_id += 1
                pred = pred_toks.get(
                    tok_id, tok if baseline_preds is None else baseline_preds[j][i]
                )
                pred_tweet.append(pred.lower())
            elif tok == "rt" and 0 < i < len(tweet) - 1 and tweet[i + 1][0] != "@":
                # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
                # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
                # of tweet and not followed by @mention) and when normalised, always to 'retweet'
                pred_tweet.append("retweet")
            else:
                pred_tweet.append(tok)
        pred_tweets.append(pred_tweet)
    if output_path is not None:
        write(raw, pred_tweets, output_path)
    return pred_tweets
