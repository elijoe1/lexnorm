from lexnorm.data.normEval import loadNormData
from lexnorm.data.baseline import write
import os
import sys


def concatenate(input1, input2, output):
    """
    concatenate two files in W-NUT 2021 format
    """
    train_raw, train_norm = loadNormData(os.path.abspath(input1))
    dev_raw, dev_norm = loadNormData(os.path.abspath(input2))
    write(
        train_raw + dev_raw,
        train_norm + dev_norm,
        os.path.join(os.path.abspath(output)),
    )


if __name__ == "__main__":
    concatenate(
        "../../../data/raw/train.norm",
        "../../../data/raw/dev.norm",
        "../../../data/interim/train.txt",
    )
