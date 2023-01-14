from lexnorm.src.data.normEval import loadNormData
from lexnorm.src.data.baseline import write
import os
from lexnorm.definitions import DATA_PATH


def concatenate():
    """
    Script to concatenate W-nut 2021 train and dev set
    """
    train_raw, train_norm = loadNormData(os.path.join(DATA_PATH, "external/train.norm"))
    dev_raw, dev_norm = loadNormData(os.path.join(DATA_PATH, "external/dev.norm"))
    write(
        train_raw + dev_raw,
        train_norm + dev_norm,
        os.path.join(DATA_PATH, "interim/train.txt"),
    )


if __name__ == "__main__":
    concatenate()
