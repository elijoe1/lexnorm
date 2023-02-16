import pandas as pd

from lexnorm.generate_extract.filtering import is_eligible


def prep_train(annotated_data):
    train = pd.read_csv(annotated_data, index_col=0)
    train_X = train.fillna(0).drop(
        ["correct", "gold", "process", "tweet", "tok"], axis=1
    )
    train_y = train.fillna(0)["correct"]
    return train_X, train_y


def prep_test(unannotated_data):
    train = pd.read_csv(unannotated_data, index_col=0)
    test_X = train.fillna(0).drop(["process", "tweet", "tok"], axis=1)
    return test_X


def normalise(raw, pred_toks):
    pred_tokens_iter = iter(pred_toks)
    pred_tweets = []
    for tweet in raw:
        pred_tweet = []
        for i, tok in enumerate(tweet):
            if is_eligible(tok):
                pred_tweet.append(next(pred_tokens_iter))
            elif tok == "rt" and 0 < i < len(tweet) - 1 and tweet[i + 1][0] != "@":
                # hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
                # we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
                # of tweet and not followed by @mention) and when normalised, always to 'retweet'
                pred_tweet.append("retweet")
            else:
                pred_tweet.append(tok)
        pred_tweets.append(pred_tweet)
    return pred_tweets
