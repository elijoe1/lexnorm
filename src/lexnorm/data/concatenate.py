from lexnorm.data.normEval import loadNormData
from lexnorm.data.baseline import write
from lexnorm.definitions import DATA_PATH
import os


def concatenate(input1, input2, output):
    """
    Concatenate two files in W-NUT 2021 format
    @param input1: file 1
    @param input2: file 2
    @param output: output file
    """
    train_raw, train_norm = loadNormData(os.path.join(DATA_PATH, input1))
    dev_raw, dev_norm = loadNormData(os.path.join(DATA_PATH, input2))
    write(
        train_raw + dev_raw,
        train_norm + dev_norm,
        os.path.join(DATA_PATH, output),
    )


if __name__ == "__main__":
    concatenate(
        "raw/train.norm",
        "raw/dev.norm",
        "interim/train.txt",
    )
