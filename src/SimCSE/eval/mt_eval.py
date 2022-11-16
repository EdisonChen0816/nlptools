# encoding=utf-8
from eval.index import Index
from eval.sim_cse import SimCSE


def eval():
    n = 0
    s = 0
    t = 0
    config = {
        'simcse_model_path': './saved_model',
        'max_len': 32
    }
    model = SimCSE(config)
    index = Index(model, './data/kg/')
    with open('./data/mt/dev.txt') as f:
        for line in f:
            query, faq, label = line.replace('\n', '').split('\t')
            if label == '0':
                continue
            vec = model.get_sentence_vec(query)
            datas = index.search(vec, 30)
            cands = model.get_score(query, datas)
            cand = sorted(cands, key=lambda x: -x[1])[0]
            score = float(cand[1])
            pred_faq = cand[0].faq
            n += 1
            if score > 0.7:
                if faq == pred_faq:
                    s += 1
                else:
                    t += 1
    r = s / n
    p = s / (s + t)
    f1 = 2 * (p * r) / (p + r)
    return f1



