import os

import pandas as pd

from lexnorm.definitions import DATA_PATH


def train():
    ...


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(DATA_PATH, "hpc/train_annotated.txt"), index_col=0)
    X_train = train.fillna(0).drop([""])
