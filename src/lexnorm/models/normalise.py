import pandas as pd

from lexnorm.generate_extract.filtering import is_eligible


def prep_train(annotated_data):
    # na_values specified as pandas would otherwise detect candidate 'NaN' as NaN rather than keeping as the string
    train = pd.read_csv(
        annotated_data, index_col=0, keep_default_na=False, na_values=""
    )
    train_X = train.drop(
        [
            "correct",
            "raw",
            "gold",
            "process",
            "tweet",
            "tok",
        ],
        axis=1,
    )
    train_y = train.fillna(0)["correct"]
    train_X.fillna(0, inplace=True)
    return train_X, train_y


def prep_test(unannotated_data):
    test = pd.read_csv(
        unannotated_data, index_col=0, keep_default_na=False, na_values=""
    )
    test_X = test.drop(
        [
            "raw",
            "process",
            "tweet",
            "tok",
        ],
        axis=1,
    )
    test_X.fillna(0, inplace=True)
    return test_X


def normalise(raw, pred_toks):
    pred_tokens_iter = iter(pred_toks)
    pred_tweets = []
    for tweet in raw:
        pred_tweet = []
        for i, tok in enumerate(tweet):
            if is_eligible(tok):
                pred = next(pred_tokens_iter)
                pred_tweet.append(pred.lower())
            elif tok == "rt" and 0 < i < len(tweet) - 1 and tweet[i + 1][0] != "@":
                # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
                # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
                # of tweet and not followed by @mention) and when normalised, always to 'retweet'
                pred_tweet.append("retweet")
            else:
                pred_tweet.append(tok)
        pred_tweets.append(pred_tweet)
    return pred_tweets
