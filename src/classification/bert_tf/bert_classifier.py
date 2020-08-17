# encoding=utf-8
import os
import glob
import pickle
import collections
import tensorflow as tf
from src.classification.bert_tf.component import Component
from src.classification.bert_tf import modeling, optimization, tokenization
from src.classification.bert_tf.bert_utils import InputFeatures, Processor


def create_model(bert_config, is_training, input_ids, input_mask, num_labels, labels=None):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        use_one_hot_embeddings=False)
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels],
        initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        if labels is None:
            return probabilities
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate=5e-5, num_train_steps=0,
                     num_warmup_steps=0):
    def model_fn(features, labels, mode, params):
        input_ids, input_mask, label_ids = [features.get(k) for k in \
                                            ("input_ids", "input_mask", "label_ids")]
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, num_labels, label_ids)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            accu = tf.metrics.accuracy(labels=label_ids, predictions= \
                tf.argmax(logits, axis=-1, output_type=tf.int32))
            loss = tf.metrics.mean(values=per_example_loss)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                                     eval_metric_ops={"eval_accu": accu, "eval_loss": loss})
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": probabilities})
        return output_spec
    return model_fn


def convert_single_example(ex_index, example, label_list, max_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens = tokenizer.tokenize(example.text)
    tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(tokens))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("label: %s(id=%d)" % (example.label, label_id))
    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        label_id=label_id)


def file_based_convert_examples_to_features(examples, label_list, max_length, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list, max_length, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature([feature.label_id])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64)}

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat().shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        return d

    return input_fn


def dump_model_fn_builder(bert_config, num_labels, init_checkpoint):
    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        proba = create_model(bert_config, False, input_ids, input_mask, num_labels)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        export_outputs = {
            'predict': tf.estimator.export.PredictOutput(proba)
        }
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=proba, export_outputs=export_outputs)
        return output_spec
    return model_fn


def serving_input_receiver_fn(max_length):
    input_ids = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_mask")
    features = {"input_ids": input_ids, "input_mask": input_mask}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


class BertClassifier(Component):

    defaults = {
        "max_length": 128,
        "batch_size": 32,
        "do_lower_case": True,
        "warmup_ratio": .1,
        "learning_rate": 2e-5,
        "epochs": 3,
        "save_checkpoints_steps": 100}

    def __init__(self, inputs, outputs, config=None):
        super(BertClassifier, self).__init__(inputs, outputs, config)
        self.tokenizer = None
        self.model = None
        self.predictor = None

    def _fit(self, data_path, bert_path, save_dirn, max_length, batch_size, do_lower_case,
             warmup_ratio, learning_rate, epochs, save_checkpoints_steps, **kwargs):
        tf.gfile.MakeDirs(save_dirn)
        processor = Processor()
        train_examples, labels = processor.get_train_examples(data_path)
        num_train_steps = int(len(train_examples) / batch_size * epochs)
        num_warmup_steps = 0 if self.model else int(num_train_steps * warmup_ratio)
        if not self.model:
            self.config["labels"] = labels
            init_checkpoint = os.path.join(bert_path, "bert_model.ckpt")
            bert_config_file = os.path.join(bert_path, "bert_config.json")
            bert_config = modeling.BertConfig.from_json_file(bert_config_file)
            model_fn = model_fn_builder(
                bert_config=bert_config,
                num_labels=len(labels),
                init_checkpoint=init_checkpoint,
                learning_rate=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
            session_config = tf.ConfigProto(log_device_placement=True)
            session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
            os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
            run_config = tf.estimator.RunConfig(
                model_dir=save_dirn,
                save_checkpoints_steps=save_checkpoints_steps,
                session_config=session_config)
            self.model = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config)
        vocab_file = os.path.join(bert_path, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        train_file = os.path.join(save_dirn, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, labels, max_length, self.tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_length,
            is_training=True,
            drop_remainder=True,
            batch_size=batch_size)
        self.model.train(input_fn=train_input_fn, max_steps=num_train_steps)
        with open(os.path.join(save_dirn, "config"), "wb") as out:
            pickle.dump(self.config, out)

    def fit(self, data_path, **kwargs):
        # TBD: download bert_tf into bert_path and set default save_dirn
        self.config = {**self.config, **kwargs}
        self._fit(data_path, **self.config)

    def _evaluate(self, data_path, bert_path, save_dirn, labels,
                  max_length, batch_size, do_lower_case, **kwargs):
        if not os.path.isdir(save_dirn):
            os.makedirs(save_dirn)
        processor = Processor()
        eval_examples, _ = processor.get_test_examples(data_path)
        vocab_file = os.path.join(bert_path, "vocab.txt")
        eval_file = os.path.join(save_dirn, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, labels, max_length, self.tokenizer, eval_file)
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_length,
            is_training=False,
            drop_remainder=False,
            batch_size=batch_size)
        return self.model.evaluate(input_fn=eval_input_fn, steps=None)

    def evaluate(self, data_path, **kwargs):
        assert self.tokenizer and self.model, "please fit model first"
        return self._evaluate(data_path, **self.config)

    def _process(self, text, bert_path, max_length, labels, **kwargs):
        vocab_file = os.path.join(bert_path, "vocab.txt")
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (max_length - len(input_ids))
        input_mask += [0] * (max_length - len(input_mask))
        features = {"input_ids": [input_ids], "input_mask": [input_mask]}
        return self.predictor(features)["output"].tolist()[0]

    def process(self, message, *args, **kwargs):
        # TBD: support serving
        assert self.predictor is not None, "please load model first"
        text = message.get(self.inputs.get("text", "text"), "")
        probs = self._process(text, **self.config)
        intent = [{"intent": name, "prob": prob} for name, prob in zip(self.config["labels"], probs)]
        message.set(self.outputs.get("intent", "intent"), intent)
        return message

    def can_process(self, message):
        return self.inputs.get("text", "text") in message

    def load(self, dirn):
        assert os.path.exists(dirn)
        with open(os.path.join(dirn, "config"), "rb") as fin:
            self.config = pickle.load(fin)
        bert_path = self.config["bert_path"]
        vocab_file = os.path.join(bert_path, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=self.config["do_lower_case"])
        saved_model = sorted(glob.glob(os.path.join(dirn, "exported", "*")))[-1]
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model)

    def _save(self, dirn, bert_path, save_dirn, max_length, labels, **kwargs):
        with open(os.path.join(dirn, "config"), "wb") as out:
            pickle.dump(self.config, out)
        bert_config_file = os.path.join(bert_path, "bert_config.json")
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        predictor = tf.estimator.Estimator(
            model_fn=dump_model_fn_builder(
                bert_config=bert_config,
                num_labels=len(labels),
                init_checkpoint=save_dirn),
            config=tf.estimator.RunConfig(model_dir=save_dirn))
        predictor.export_savedmodel(os.path.join(dirn, "exported"),
                                    serving_input_receiver_fn(max_length))

    def save(self, dirn):
        # save to serving models
        assert self.config and self.model, "please fit model first"
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        self._save(dirn, **self.config)


if __name__ == "__main__":
    pass
