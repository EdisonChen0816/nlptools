# encoding=utf-8
from src.classification.bert_tf.message import Message
import json
import os
import shutil
from src.classification.bert_tf.bert_classifier import BertClassifier


def test_bert_tf_classifier(message):
    data_path = '../../../data/bert/atis-train.iob'
    test_path = '../../../data/bert/atis-dev.iob'
    bert_path = 'C:/数据/模型/chinese_L-12_H-768_A-12'
    save_path = '../../../model/bert'
    bert_cfg = {
        "bert_path": bert_path,
        "max_length": 128,
        "batch_size": 32,
        "save_path": save_path
    }
    init_cfg = {
        "inputs": {
            "text": "text"
        },
        "outputs": {
            "intent": "intent"
        },
        "config": None
    }
    train_cfg = {
        "data_path": data_path,
        "learning_rate": 2e-5,
        "epochs": 3,
        "save_checkpoints_steps": 1000
    }
    init_cfg["config"] = bert_cfg
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    model = BertClassifier(**init_cfg)
    model.fit(**train_cfg)
    print(model.evaluate(data_path=test_path))
    model.save("models")
    model = BertClassifier(**init_cfg)
    model.load("models")
    message = model.process(message)
    print(json.dumps(message.get(init_cfg["outputs"]["intent"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    message = Message("what are the flights and fares from atlanta to philadelphia")
    test_bert_tf_classifier(message)