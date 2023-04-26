import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from lexnorm.generate_extract.filtering import is_eligible
from lexnorm.data.baseline import write


def load_candidates(candidates_path, random_state=None, shuffle=False):
    """
    Opens candidates dataframe produced from process_data correctly

    :param random_state: Random state for shuffle
    :param candidates_path: Path to candidates dataframe saved in csv format
    :param shuffle: Whether to shuffle the dataframe rows (avoids various issues)
    :return: Candidates dataframe
    """
    candidates_df = pd.read_csv(
        candidates_path, index_col=0, keep_default_na=False, na_values=""
    )
    if shuffle:
        candidates_df = candidates_df.sample(
            frac=1,
            random_state=random_state,
        )
    return candidates_df


def prep_train(annotated_dataframe):
    train_X = annotated_dataframe.drop(
        [
            "correct",
            "raw",
            "gold",
            "prev",
            "next",
            "tok_id",
        ],
        axis=1,
    )
    train_X.spellcheck_rank = train_X.spellcheck_rank.fillna(23)
    train_X.embeddings_rank = train_X.embeddings_rank.fillna(11)
    train_X.cosine_to_orig = train_X.cosine_to_orig.fillna(-1)
    train_X = train_X.fillna(0)
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    # train_X = imp_mean.fit_transform(train_X)
    train_y = annotated_dataframe.fillna(0)["correct"]
    return train_X, train_y


def prep_test(unannotated_dataframe):
    test_X = unannotated_dataframe.drop(
        [
            "raw",
            "prev",
            "next",
            "tok_id",
        ],
        axis=1,
    )
    # Spylls has well-defined suggestion limit:
    # 15 edit based + 3 compound + 4 ngram based = 22
    test_X.spellcheck_rank = test_X.spellcheck_rank.fillna(23)
    # max 10 embeddings returned
    test_X.embeddings_rank = test_X.embeddings_rank.fillna(11)
    # if cannot calculate cosine similarity, make as dissimilar as possible
    test_X.cosine_to_orig = test_X.cosine_to_orig.fillna(-1)
    # where filling with 0 is the intended behaviour
    test_X = test_X.fillna(0)
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    # test_X = imp_mean.fit_transform(test_X)
    return test_X


def normalise(raw, pred_toks: dict, output_path=None, baseline_preds=None):
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
