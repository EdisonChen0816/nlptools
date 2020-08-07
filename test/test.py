# encoding=utf-8
from src.similarity.text_sim import get_score, q2v
from gensim.models import KeyedVectors
import jieba.analyse

w2v = KeyedVectors.load('./model/w2v/w2v.model')

TP = 0  #预测为正，真实为正
FP = 0  #预测为正，真实为负
FN = 0  #预测为负，真实为正
TN = 0  #预测为负，真实为负


with open('c:/测试数据.txt', 'r', encoding='utf-8') as f:
    for line in f:
        s1, s2, label = line.replace('\n', '').split('\t')
        label = int(label)
        qws = {}
        tws = {}
        qts = []
        tts = []
        for k, v in jieba.analyse.extract_tags(s1, topK=10, withWeight=True):
            qws[k] = v
            qts.append(k)
        for k, v in jieba.analyse.extract_tags(s2, topK=10, withWeight=True):
            tws[k] = v
            tts.append(k)
        qv = q2v(tws, w2v)
        score = get_score(qts, tts, qws, tws, w2v, qv)
        score -= 2.0
        if score > 0.0:
            pred = 1
        else:
            pred = 0
        if 1 == pred and 1 == label:
            TP += 1
        elif 1 == pred and 0 == label:
            FP += 1
        elif 0 == pred and 1 == label:
            FN += 1
        elif 0 == pred and 0 == label:
            TN += 1
acc = (TP + TN) / 100
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = (2 * precision * recall) / (precision + recall)


#acc:0.94 precision:0.9074074074074074 recall:0.98 f1:0.9423076923076924
print('acc:' + str(acc), 'precision:' + str(precision), 'recall:' + str(recall), 'f1:' + str(f1))