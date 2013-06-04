"""
process:
    input features
    train the model
    predict scores for each test record and output the order
"""
import random
import numpy as np

from core import LambdaRank , DataSpace
from config import (TRAINSET_PAIR_PATH, 
        TRAINSET_USER_FEATURE_PATH_, 
        TRAINSET_ITEM_FEATURE_PATH,
        TRAIN_SET_NUM,
        )
from utils import show_status, data_path

class DataInputer(object):
    """
    read and process dataset from dataset file

    user_features:
    {
        userid: "features splited by space",
        1: "2.3 2.1 4.2 ...",
    }

    item_features:
        similar to user features

    pairs:
        list of pairs 
        [
            "authorid positive-recordid negitive-recordid",
            "3423 23 3434",
        ]
    """
    def __init__(self):
        self.user_features = {}
        self.item_features = {}
        self.pairs = []

    def __call__(self):
        self.input_user_features()
        self.input_item_features()
        self.input_pair_features()
        self.split_dataset()

    def get_dataset(self, i):
        """
        iter to yield data

        ("userfeature positive-record-features" ,
            "userfeature negtive-record-features")
        """
        for pair in self.data_sets[i]:
            userid, pid, nid = [int(p) for p in pair.split()]
            record = (' '.join(self.user_features[userid], self.item_features[pid]), 
            ' '.join(self.user_features[userid], self.item_features[nid]))
            yield record

    def input_user_features(self):
        show_status(".. input user features")
        with open(TRAINSET_USER_FEATURE_PATH_) as f:
            for line in f.readlines():
                ws = line.split()
                userid = int(ws[0])
                features = ws[1:]
                self.item_features[userid] = ' '.join(features)

    def input_item_features(self):
        show_status(".. input item features")
        with open(TRAINSET_ITEM_FEATURE_PATH) as f:
            for line in f.readlines():
                ws = line.split()
                itemid = int(ws[0])
                features = ws[1:]
                self.item_features[itemid] = ' '.join(features)

    def input_pair_features(self):
        show_status(".. input pairs features")
        with open(TRAINSET_PAIR_PATH) as f:
            for line in f.readlines():
                self.pairs.append(line)
            show_status(".. random shuffle pairs")
            random.shuffle(self.pairs)

    def split_dataset(self):
        """
        spit dataset to several splits
        and create a validation dataset
        """
        show_status(".. split dataset to %d pieces" % TRAIN_SET_NUM)
        num = len(self.pairs)
        piece_len = int(TRAIN_SET_NUM / num)
        index = 0
        self.data_sets = [ self.pairs[index : index + piece_len] for i in xrange(TRAIN_SET_NUM)]
        del self.pairs



class Trainer(object):
    def __init__(self, dataset):
        """
        dataset: object of DataInputer
        """
        self.dataset = dataset
        self.model = LambdaRank()

    def run(self):
        """
        record is a string line
        """
        val_maps = []
        for val_idx in xrange(TRAIN_SET_NUM):
            # user ith dataset as a validate dataset
            set_indexs = set(range(TRAIN_SET_NUM))
            set_indexs.discard(val_idx)
            self.train(set_indexs)
            val_res = self.validate()
            val_maps.append(val_res)
        map_res = sum(val_maps) / TRAIN_SET_NUM
        self.model.dataspace.tofile(data_path('models', str(map_res)))

    def train(self, set_indexs):
        # train using the rest dataset
        for i in list(set_indexs):
            for i, (X1, X2) in enumerate(self.dataset.get_dataset(1)): 
                X1 = np.array([float(i) for i in X1.split()])
                X2 = np.array([float(i) for i in X2.split()])
                self.model.study_line(X1, X2)


    def validate(self, val_set):
        """
        validate and save best MAP
        """
        # TODO how to validate?
        pass
