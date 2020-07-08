# encoding=utf8
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from gensim.models import KeyedVectors


def BM25(q, t):
    k = 1.5
    b = 0.75
    score = 0.0
    sq = 0.0
    for qt in q:
        if qt in t:
            score += q[qt] * t[qt] * (k + 1) / (t[qt] + k * (1 - b + b * len(t) / 10))
        sq += q[qt]
    if sq > 0:
        s = score / sq
        if s > 0.2:
            s = 0.2 + (s - 0.2) * 0.6
        return s
    else:
        return -1


def ctr(q, t):
    s1 = 0.0
    s2 = 0.0
    for qt in q:
        s1 += q[qt]
        if qt in t:
            s2 += t[qt]
    if s1 == 0:
        return 0.0
    else:
        s = s2 / s1
        if s > 0.5:
            s = 0.5 + (s - 0.5) * 0.5
        return s


def cqr(q, t):
    s1 = 0.0
    s2 = 0.0
    for qt in t:
        s1 += t[qt]
        if qt in q:
            s2 += q[qt]
    if s1 == 0:
        return 0.0
    else:
        s = s2 / s1
        if len(q) - len(t) > 0:
            s /= math.sqrt(1 + (len(q) - len(t)))
        if s > 0.5:
            s = 0.5 + (s - 0.5) * 0.4
    return s


def wmd(s1, s2, w2v):
    return math.sqrt(w2v.wmdistance(s1, s2)) * 0.5


def se_sim(qv, tw, w2v):
    fv = np.zeros([1, 300]).astype('float32')
    for term in tw:
        if term in w2v:
            fv += np.array(w2v[term])
    return cosine_similarity(qv, fv)[0][0]


def sim_hist(s1, s2, qw, w2v):
    s1 = filter(lambda x: x in w2v, s1)
    s2 = filter(lambda x: x in w2v, s2)
    hist = [0.0] * 10
    for i in range(len(s1)):
        w1 = 1.0
        if s1[i] in qw:
            w1 = qw[s1[i]]
        w2 = 0.0
        if s1[i] in qw:
            w2 += qw[s1[i]]
        else:
            w2 += 1.0
        if i + 1 < len(s1):
            if s1[i + 1] in qw:
                w2 += qw[s1[i + 1]]
            else:
                w2 += 1.0

        for j in range(len(s2)):
            d = w2v.distance(s1[i], s2[j])
            if d == 1:
                hist[0] += 1.0 * w1
            elif d > 0.95:
                hist[1] += 1.0 * w1
            elif d > 0.85:
                hist[2] += 1.0 * w1
            elif d > 0.6:
                hist[3] += 1.0 * w1
            else:
                hist[4] += 1.0 * w1

            if i < len(s1) - 1 and j < len(s2) - 1:
                d2 = w2v.n_similarity(s1[i:i + 2], s2[j:j + 2])
                if d2 == 1:
                    hist[5] += 1.0 * w2
                elif d2 > 0.9:
                    hist[6] += 1.0 * w2
                elif d2 > 0.75:
                    hist[7] += 1.0 * w2
                elif d2 > 0.5:
                    hist[8] += 1.0 * w2
                else:
                    hist[9] += 1.0 * w2
    for i in range(len(hist)):
        if hist[i] > 0:
            hist[i] = math.log(hist[i])
    return hist


def query_len_penalty(q):
    return math.log(len(q), 4) - 1.5


def title_len_penalty(t):
    return math.log(len(t)+2, 4) - 1.3


def q2v(tws, w2v):
    v = np.zeros([1, 300]).astype('float32')
    for k in tws:
        if k in w2v:
            v += np.array(w2v[k]) * tws[k]
    if len(tws) > 0:
        v /= float(len(tws))
    return v


def get_score(qts, tts, qws, tws, w2v, qv):
    return BM25(qws, tws) * 2 + ctr(qws, tws) + cqr(qws, tws) - wmd(qts, tts, w2v) + se_sim(qv, tts, w2v)[0] + query_len_penalty(qts) + title_len_penalty(tts) - 2.0


if __name__ == '__main__':
    import jieba.analyse
    s1 = '今天天气怎么样'
    s2 = '今天天气如何'
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
    w2v = KeyedVectors.load('../../model/w2v/w2v.model')
    print(BM25(qws, tws))
    print(ctr(qws, tws))
    print(cqr(qws, tws))
    print(query_len_penalty(s1))
    print(title_len_penalty(s2))
    print(wmd(qts, tts, w2v))
    print(se_sim(q2v(tws, w2v), tts, w2v))



