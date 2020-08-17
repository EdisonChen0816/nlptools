# encoding=utf-8
import random
from src.classification.bert_tf import tokenization


class InputExample(object):

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class Processor(object):

    def __init__(self):
        pass

    def get_train_examples(self, data_path):
        return self._create_examples(data_path, "train")

    def get_test_examples(self, data_path):
        return self._create_examples(data_path, "test")

    def _create_examples(self, data_path, set_type):
        labels, examples = set(), []
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                label, message = line.strip().split("\t")
                if "#" in label:
                    continue
                labels.add(label)
                guid = "{0}-{1}-{2}".format(set_type,label,i)
                text = tokenization.convert_to_unicode(message)
                label = tokenization.convert_to_unicode(label)
                examples.append(InputExample(guid=guid,text=text,label=label))
        random.shuffle(examples)
        return examples, list(labels)