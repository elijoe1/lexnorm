import pandas as pd

from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.data.baseline import write


def prep_train(annotated_dataframe):
    # na_values specified as pandas would otherwise detect candidate 'NaN' as NaN rather than keeping as the string
    train_X = annotated_dataframe.drop(
        [
            "correct",
            "raw",
            "gold",
            "prev",
            "next",
            "process",
            "tweet",
            "tok",
            # "twitter_uni",
            # "twitter_bi_prev",
            # "twitter_bi_next",
            # "wiki_uni",
            # "wiki_bi_prev",
            # "wiki_bi_next",
        ],
        axis=1,
    ).fillna(0)
    train_y = annotated_dataframe.fillna(0)["correct"]
    return train_X, train_y


def prep_test(unannotated_dataframe):
    test_X = unannotated_dataframe.drop(
        [
            "raw",
            "prev",
            "next",
            "process",
            "tweet",
            "tok",
            # "twitter_uni",
            # "twitter_bi_prev",
            # "twitter_bi_next",
            # "wiki_uni",
            # "wiki_bi_prev",
            # "wiki_bi_next",
        ],
        axis=1,
    )
    test_X.fillna(0, inplace=True)
    return test_X


def normalise(raw, pred_toks: dict, output_path=None):
    tok_id = -1
    pred_tweets = []
    for tweet in raw:
        pred_tweet = []
        for i, tok in enumerate(tweet):
            if is_eligible(tok):
                tok_id += 1
                pred = pred_toks.get(tok_id, tok)
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
