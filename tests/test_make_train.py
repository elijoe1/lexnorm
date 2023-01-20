from lexnorm.data.normEval import loadNormData
from lexnorm.data import make_train
from lexnorm.definitions import DATA_PATH
import os


def test_concatenate(tmp_path):
    make_train.concatenate(
        "raw/train.norm", "raw/dev.norm", os.path.join(tmp_path, "concatenated.txt")
    )
    concatenated = loadNormData(os.path.join(tmp_path, "concatenated.txt"))
    train = loadNormData(os.path.join(DATA_PATH, "raw/train.norm"))
    dev = loadNormData(os.path.join(DATA_PATH, "raw/dev.norm"))
    join = (train[0] + dev[0], train[1] + dev[1])
    assert concatenated == join
    assert len(concatenated[0]) == len(train[0]) + len(dev[0])
