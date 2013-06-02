"""
process:
    input features
    train the model
    predict scores for each test record and output the order
"""
from core import LambdaRank 
from config import TRAINSET_PAIR_PATH, TRAINSET_USER_FEATURE_PATH_, TRAINSET_ITEM_FEATURE_PATH

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

    def input_user_features(self):
        with open(TRAINSET_USER_FEATURE_PATH_) as f:
            for line in f.readlines():
                ws = line.split()
                userid = int(ws[0])
                features = ws[1:]
                self.item_features[userid] = ' '.join(features)

    def input_item_features(self):
        with open(TRAINSET_ITEM_FEATURE_PATH) as f:
            for line in f.readlines():
                ws = line.split()
                itemid = int(ws[0])
                features = ws[1:]
                self.item_features[itemid] = ' '.join(features)

    def input_pair_features(self):
        with open(TRAINSET_PAIR_PATH) as f:
            for line in f.readlines():
                # TODO to randomly order it?
                self.pairs.append(line)

    @property
    def data(self):
        """
        iter to yield data

        ("userfeature positive-record-features" ,
            "userfeature negtive-record-features")
        """
        for pair in self.pairs:
            userid, pid, nid = [int(p) for p in pair.split()]
            record = (' '.join(self.user_features[userid], self.item_features[pid]), 
            ' '.join(self.user_features[userid], self.item_features[nid]))
            yield record


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
        for i, (X1, X2) in enumerate(self.dataset.data):
            X1 = X1.split()
            X2 = X2.split()
            self.model.study_line(X1, X2)
