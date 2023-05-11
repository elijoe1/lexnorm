import pandas as pd


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
    ).fillna(0)
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
    ).fillna(0)
    return test_X
