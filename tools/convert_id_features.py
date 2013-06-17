"""
preappend uid, itemid to user features and item features
"""
import json
import sys
sys.path.append('..')
from utils import show_status, data_path, get_path
from config import Config

class IDConverter(object):
    def __init__(self, id_json_path, feature_path, transed_features_path):
        self.id_json_path, self.feature_path = id_json_path, feature_path
        self.transed_features_path = transed_features_path
        self.id_line_no_dic = {}
        self.features = []

    def load_id_json(self):
        with open(self.id_json_path) as f:
            content = f.read()
            self.id_line_no_dic = json.loads(content)

    def load_features(self):
        with open(self.feature_path) as f:
            for line in f.readlines():
                self.features.append(line)

    def trans(self):
        def preappend(id, line):
            line = "%d %s" % (int(id), line)
            return line

        for key,value in self.id_line_no_dic.items():
            no = int(value)
            self.features[no] = preappend(key, self.features[no])

    def tofile(self):
        show_status(".. to file: %s" % self.transed_features_path)
        with open(self.transed_features_path, 'w') as f:
            f.write(''.join(self.features))

    def __call__(self):
        self.load_id_json()
        self.load_features()
        self.trans()
        self.tofile()

def trans():
    user_converter = IDConverter(
            '../data/author_map.json',
            '../data/user_features.txt',
            Config.TRAINSET_USER_FEATURE_PATH
        )
    user_converter()

    item_converter = IDConverter(
            '../data/paper_map.json',
            '../data/item_features.txt',
            Config.TRAINSET_ITEM_FEATURE_PATH
        )
    item_converter()

if __name__ == '__main__':
    trans()
