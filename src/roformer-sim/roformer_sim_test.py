# encoding=utf-8
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
import faiss
import tensorflow as tf



maxlen = 32
# bert配置
config_path = '/Users/edison/Documents/software/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/edison/Documents/software/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/edison/Documents/software/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)
# 加载rofomer-sim模型
encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])


# h5模型转pb模型
def keras_model_to_tfs(model, export_path):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'token_ids': model.input[0], 'segment_ids': model.input[1]},
        outputs={'output': model.output}
    )
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    print('Build done.')


# keras_model_to_tfs(encoder, '../model/test/1')


def similarity(text1, text2):
    """"计算text1与text2的相似度
    todo 目前是一个一个计算，可以改成批量计算
    """
    texts = [text1, text2]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return (Z[0] * Z[1]).sum()


id2name = dict()
with open('../data/标准问.txt') as f:
    for line in f:
        id, name, _ = line.replace('\n', '').split('\t')
        id2name[id] = name
id2name['-20000'] = '负样本'


class Item:

    def __init__(self, id, sim_query, tq_id, tq_name):
        self.id = id
        self.sim_query = sim_query
        self.tq_id = tq_id
        self.tq_name = tq_name


# query -> roformer-sim向量
def q2v(query):
    X, S = [], []
    x, s = tokenizer.encode(query, maxlen=maxlen)
    X.append(x)
    S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    return encoder.predict([X, S])


def get_vectors_and_mapping():
    id2item = dict()
    count = 0
    X, S = [], []
    with open('../data/faiss/test1/train.txt') as f:
        for line in f:
            try:
                query, tq_id, _ = line.replace('\n', '').split('\t')
                item = Item(count, query, tq_id, id2name[tq_id])
                id2item[count] = item
                count += 1
                x, s = tokenizer.encode(query, maxlen=maxlen)
                X.append(x)
                S.append(s)
            except:
                print(line)
    X = sequence_padding(X)
    S = sequence_padding(S)
    vecs = encoder.predict([X, S])
    # todo vecs可以save
    return vecs, id2item


vectors, id2item = get_vectors_and_mapping()
index = faiss.index_factory(768, 'Flat', faiss.METRIC_L2)
index.add(vectors)

# f1 = open('/Users/edison/Desktop/roformer_测试结果.txt', 'a+', encoding='utf-8')
cnt_top1 = 0
cnt_top3 = 0
with open('../data/faiss/test1/test1.txt') as f:
    for line in f:
        cands = []
        test_query, tq_id, tq_name = line.replace('\n', '').split('\t')
        test_v = q2v(test_query)
        _, I_30 = index.search(test_v, 30)
        a_token_ids, b_token_ids = [], []
        for id in I_30[0]:
            if id in [-1, '-1']:
                continue
            item = id2item[id]
            sim_query = item.sim_query
            score = similarity(test_query, sim_query).item()
            cands.append((item, score))
        cands = sorted(cands, key=lambda x: -x[1])
        if cands[0][0].tq_name.strip() == tq_name.strip() and cands[0][1] > 0.85:
            cnt_top1 += 1
        for cand in cands[:3]:
            if cand[0].tq_name.strip() == tq_name.strip() and cand[1] > 0.85:
                cnt_top3 += 1
                break
        # f1.writelines(test_query + '\t' + tq_name + '\t' + cands[0][0].tq_name + '\t' + cands[0][0].sim_query + '\t' + str(cands[0][1]) + '\t'
        #               + cands[1][0].tq_name + '\t' + cands[1][0].sim_query + '\t' + str(cands[1][1]) + '\t' + cands[2][0].tq_name + '\t'
        #               + cands[2][0].sim_query + '\t' + str(cands[2][1]) + '\n')


print(str(cnt_top1) + '\t' + str(cnt_top3))
# f1.close()