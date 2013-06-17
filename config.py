import os
DIRNAME = "/infomall/hist/chunwei/lambdarank/data"
# split trainset to num pieces
from utils import data_path
TRAIN_SET_NUM = 5


class Config(object):
# total feature num of an item
    K = 200
    J = 10
# learning rate
    THETA = 0.01
    SIGMA = 0.01

TRAINSET_PAIR_PATH = os.path.join(DIRNAME, "Train.csv")
TRAINSET_USER_FEATURE_PATH = os.path.join(DIRNAME, "author_features.chun")
TRAINSET_ITEM_FEATURE_PATH = os.path.join(DIRNAME, "item_features.chun")
TRAINSET_VIRTUAL_MEM_DIR = os.path.join(DIRNAME, "trainset")

MAX_PAIRS_SINGLE_LINE = 500
