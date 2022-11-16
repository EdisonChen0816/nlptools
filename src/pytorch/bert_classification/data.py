# encoding=utf-8
import torch
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/edison.chen/models/MTBert_v0.0.1')
# tokenizer = AutoTokenizer.from_pretrained('/Users/edison/Documents/工作/数据和模型/模型/bert-base-chinese')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.labels = []
        self.texts = []
        with open(data_path) as f:
            for line in f:
                cols = line.replace('\n', '').split('\t')
                if len(cols) != 3:
                    continue
                sen1, sen2, label = cols
                self.labels.append(int(label))
                self.texts.append(
                    tokenizer(sen1 + '[SEP]' + sen2, padding='max_length', max_length=32, truncation=True, return_tensors='pt')
                )

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
