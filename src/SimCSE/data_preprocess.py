# encoding=utf-8
import pickle
from collections import OrderedDict

f1 = open('./data/mt_sim/sup_train.txt', 'a+', encoding='utf-8')
res = open('./data/resutls_20221107.pkl', 'rb')
datas = pickle.load(res)
for query in datas:
    sim_datas = datas[query]
    query = query.replace('\t', ' ')
    sim_list = []
    hard_neg = []
    for i in range(3):
        try:
            for sd in sim_datas[i]:
                sd = sd.replace('\t', ' ')
                sim_list.append(sd)
        except:
            pass
    try:
        for sd in sim_datas[20]:
            sd = sd.replace('\t', ' ')
            hard_neg.append(sd)
    except:
        pass
    try:
        for sd in sim_datas[70]:
            sd = sd.replace('\t', ' ')
            hard_neg.append(sd)
    except:
        pass
    if len(sim_list) != 0 and len(hard_neg) != 0:
        for sim in sim_list:
            for he in hard_neg:
                f1.writelines(query + '\t' + sim + '\t' + he + '\n')

f1.close()


