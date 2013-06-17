"""
process:
    input features
    train the model
    predict scores for each test record and output the order
"""
import os
import random
import numpy as np

from core import LambdaRank , DataSpace
from config import (TRAINSET_PAIR_PATH, 
        TRAINSET_USER_FEATURE_PATH, 
        TRAINSET_ITEM_FEATURE_PATH,
        TRAIN_SET_NUM, MAX_PAIRS_SINGLE_LINE,
        TRAINSET_VIRTUAL_MEM_DIR,
        )
from utils import show_status, data_path, cal_map

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
        # original raw text data
        self.trainset = []
        self.train_pairs = {}

    def __call__(self):
        self.input_user_features()
        self.input_item_features()
        self.input_trainset()
        self.split_trainset()
        self.trans_pairs()

    def get_data_line(self, uid, pid, nid=None):
        if nid is None:
            record = ' '.join(self.user_features[uid], self.item_features[pid])
        else:
            record = (' '.join((self.user_features[uid], self.item_features[pid])), 
            ' '.join((self.user_features[uid], self.item_features[nid])))
        return record

    def get_dataset(self, i):
        """
        iter to yield data

        ("userfeature positive-record-features" ,
            "userfeature negtive-record-features")
        """
        for uid, pid, nid in self.train_pairs[i]:
            record = self.get_data_line(uid, pid, nid)
            yield record

    def input_user_features(self):
        show_status(".. input user features")
        with open(TRAINSET_USER_FEATURE_PATH) as f:
            for line in f.readlines():
                ws = line.split()
                userid = int(ws[0])
                features = ws[1:]
                self.user_features[userid] = ' '.join(features)

    def input_item_features(self):
        show_status(".. input item features")
        with open(TRAINSET_ITEM_FEATURE_PATH) as f:
            for line in f.readlines():
                ws = line.split()
                itemid = int(ws[0])
                features = ws[1:]
                self.item_features[itemid] = ' '.join(features)

    def input_trainset(self):
        show_status(".. input trainset")
        with open(TRAINSET_PAIR_PATH) as f:
            for line in f.readlines():
                uid, p_papers, n_papers = line.split(',')
                self.trainset.append((uid, p_papers, n_papers))
    def input_positive_trainset(self):
        """
        only input positive trainset for validate
        """
        show_status(".. input positive trainset")
        with open(TRAINSET_PAIR_PATH) as f:
            for line in f.readlines():
                uid, p_papers, n_papers = line.split(',')
                self.trainset.append((uid, p_papers, []))

    def split_trainset(self):
        """
        spit dataset to several splits
        and create a validation dataset
        """
        show_status(".. split dataset to %d pieces" % TRAIN_SET_NUM)
        num = len(self.trainset)
        piece_len = int(num / TRAIN_SET_NUM)
        index = 0
        self.trainsets = [ self.trainset[index : index + piece_len] for i in xrange(TRAIN_SET_NUM)]

    def trans_pairs(self):
        """
        traindata: a line of self.trainset
            (uid, p_papers, n_papers)
        """
        show_status(".. trains_pairs")
        for idx in range(TRAIN_SET_NUM):
            dataset = self.trainsets[idx]
            for d in dataset:
                # train pairs
                (uid, p_papers, n_papers) = (int(d[0]), 
                            [int(i) for i in d[1].split()],
                            [int(i) for i in d[2].split()])
                pairs = [(uid, p, n) for p in p_papers for n in n_papers]
                random.shuffle(pairs)
                if len(pairs) > MAX_PAIRS_SINGLE_LINE:
                    pairs = pairs[:MAX_PAIRS_SINGLE_LINE]
                # add pairs to trains_set
                if self.train_pairs.get(idx, None) is None:
                    self.train_pairs[idx] = []
                self.train_pairs[idx] += pairs

    def tofile(self):
        """
        put trainset to files
        """
        for i in range(TRAIN_SET_NUM):
            show_status("output %th dataset" % i)
            dataset = self.dataset.get_dataset(i)
            with open(os.path.join(TRAINSET_VIRTUAL_MEM_DIR, str(i)+'.chun'), 'w') as f:
                strs = "\n".join(dataset)
                f.write(strs)

# read BATCH_SIZE one time
BATCH_SIZE = 1000

class VirtualDataset(DataInputer):
    """
    virtual memory control
    read several lines from disk per time
    """
    def trainset_file(self):
        self.input_user_features()
        self.input_item_features()
        self.input_trainset()
        self.split_trainset()
        self.trans_pairs()
        self.tofile()

    def get_dataset(self, i):
        with open(os.path.join(TRAINSET_VIRTUAL_MEM_DIR, str(i)+'.chun')) as f:
            records = []
            for i, line in enumerate(f.readlines()):
                record = line[:-1].split(",")
                records.append(record)
                if i == 0:
                    continue
                if i % BATCH_SIZE == 0:
                    show_status(".. given %dth record" % i)
                    yield records



class Trainer(object):
    def __init__(self, dataset):
        """
        dataset: object of DataInputer
        """
        self.dataset = dataset
        self.model = LambdaRank()

    def __call__(self):
        self.run()

    def run(self):
        """
        record is a string line
        """
        val_maps = []
        for val_idx in xrange(TRAIN_SET_NUM):
            # user ith dataset as a validate dataset
            self.val_idx = val_idx
            set_indexs = set(range(TRAIN_SET_NUM))
            set_indexs.discard(val_idx)

            self.train(set_indexs)
            val_res = self.validate()
            show_status(".. get map: " + str(val_res)) 
            val_maps.append(val_res)
        map_res = sum(val_maps) / TRAIN_SET_NUM
        show_status(".. get avage map: " + str(map_res)) 
        self.model.dataspace.tofile(data_path('models', str(map_res)))

    def train(self, set_indexs):
        # train using the rest dataset
        for i in list(set_indexs):
            show_status(">>>" * 20)
            show_status(".. training %dth dataset" % i)
            for dataset in self.dataset.get_dataset(i):
                dataset_len = len(self.dataset.train_pairs[i])
                show_status("dataset len: %d" % dataset_len)
                for i, (X1, X2) in enumerate(dataset): 
                    print "train %dth line" % i
                    show_status(">> training data", i, dataset_len)
                    X1 = np.array([float(i) for i in X1.split()])
                    X2 = np.array([float(i) for i in X2.split()])
                    self.model.study_line(X1, X2)


    def validate(self):
        """
        validate and save best MAP
        """
        def mysort(l1, l2):
            if l1[1] == l2[1]:
                return 0
            if l1[1] > l2[1]:
                return -1
            return 1
        # TODO how to validate?
        vali_set = self.dataset.trainset[self.val_idx]
        uid, p_papers, n_papers = vali_set.split(',')
        uid = int(uid)
        p_papers = [int(i) for i in p_papers]
        n_papers = [int(i) for i in n_papers]
        predicts = []
        for p in p_papers + n_papers:
            p_feature = self.dataset.get_data_line(uid, p)
            score = self.model.predict(p_feature)
            predicts.append((p, score))
        predicts.sort(mysort)
        return cal_map(p_papers, predicts)

def train_pairs_to_file():
    datainput = DataInputer()
    datainput()

if __name__ == '__main__':

    trainer = Trainer(datainput)
    trainer()

