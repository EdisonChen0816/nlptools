# encoding=utf-8
from annoy import AnnoyIndex
import os


def load_data_from_dir(path):
    for dir_path, dir_names, file_names in os.walk(path):
        for file in file_names:
            full_path = os.path.join(dir_path, file)
            for line in open(full_path, encoding='UTF-8'):
                yield line.strip().split("\t")


class Item:

    def __init__(self, sim_ques, faq, ans, vec):
        self.sim_ques = sim_ques
        self.faq = faq
        self.ans = ans
        self.vec = vec


class Index:

    def __init__(self, model, data_path):
        self.data_path = data_path
        self.index = {}
        self.items = {}
        self.u = AnnoyIndex(768)
        self.model = model
        self.build_annoy_index()

    def build_annoy_index(self):
        count = 0
        for item in load_data_from_dir(self.data_path):
            if len(item) == 3:
                vec = self.model.get_sentence_vec(item[0])
                self.items[count] = Item(item[0], item[1], item[2], vec)
                self.u.add_item(count, vec)
            elif len(item) == 2:
                vec = self.model.get_sentence_vec(item[0])
                self.items[count] = Item(item[0], item[0], item[1], vec)
                self.u.add_item(count, vec)
            count += 1
        self.u.build(20)

    def seach_vec(self, v, top_num=30):
        return self.u.get_nns_by_vector(v, n=top_num)

    def add(self, i):
        if i.key not in self.index:
            self.index[i.key] = []
        self.index[i.key].append(i)

    def search(self, vec, top_num):
        ids = self.seach_vec(vec, top_num)
        datas = []
        for id in ids:
            datas.append(self.items[id])
        return datas