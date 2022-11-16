# encoding=utf-8
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class SimCSE:

    def __init__(self, config):
        simcse_model_path = config['simcse_model_path']
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:7" if use_cuda else "cpu")
        # 建立分词器
        self.tokenizer = AutoTokenizer.from_pretrained(simcse_model_path)
        # 建立加载模型
        self.model = AutoModel.from_pretrained(simcse_model_path).to(self.device)
        self.max_len = int(config['max_len'])

    def get_sentence_vec(self, sentence):
        inputs = self.tokenizer([sentence], padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_len).to(self.device)
        with torch.inference_mode(mode=True):
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu().numpy()
        return embeddings[0]

    def get_score(self, query, datas):
        cands, a_vecs, b_vecs = [], [], []
        a_vec = self.get_sentence_vec(query)
        for data in datas:
            a_vecs.append(a_vec)
            b_vecs.append(data.vec)
        a_vecs = np.asarray(a_vecs)
        b_vecs = np.asarray(b_vecs)
        a_vecs = a_vecs / (a_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        b_vecs = b_vecs / (b_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
        sims = (a_vecs * b_vecs).sum(axis=1)
        for i in range(len(datas)):
            cands.append((datas[i], sims[i]))
        return cands
