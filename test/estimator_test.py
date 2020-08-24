# encoding=utf-8
import tensorflow as tf


def input_fn(filenames, batch_size=32, num_epochs=None, perform_shuffle=False):
    """
    每次调用，从TFRecord文件中读取一个大小为batch_size的batch
    Args:
        filenames: TFRecord文件
        batch_size: batch_size大小
        num_epochs: 将TFRecord中的数据重复几遍，如果是None，则永远循环读取不会停止
        perform_shuffle: 是否乱序
    Returns:
        tensor格式的，一个batch的数据
    """
    def _parse_fn(record):
        features = {
            "label": tf.FixedLenFeature([], tf.int64),
            "image": tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, features)
        # image
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [28, 28])
        # label
        label = tf.cast(parsed["label"], tf.int64)
        return {"image": image}, label

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    # ==========  解析参数部分  ========== #
    learning_rate = params["learning_rate"]

    # ==========  网络结构部分  ========== #
    # input
    X = tf.cast(features["image"], tf.float32, name="input_image")
    X = tf.reshape(X, [-1, 28 * 28]) / 255
    # DNN
    deep_inputs = X
    deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=128)
    deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=64)
    y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=10)
    # output
    y = tf.reshape(y_deep, shape=[-1, 10])
    pred = tf.nn.softmax(y, name="soft_max")

    # ==========  如果是 predict 任务  ========== #
    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ==========  如果是 eval 任务  ========== #
    one_hot_label = tf.one_hot(tf.cast(labels, tf.int32, name="input_label"), depth=10, name="label")
    # 构建损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=one_hot_label))
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(tf.math.argmax(one_hot_label, axis=1), tf.math.argmax(pred, axis=1))
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ==========  如果是 train 任务  ========== #
    # 构建train_op
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss,
                                                                                                                  global_step=tf.train.get_global_step())
    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main():
    # ==========  准备参数 ========== #
    task_type = "train"
    model_params = {
        "learning_rate": 0.001,
    }

    # ==========  构建Estimator  ========== #
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1}),
        log_step_count_steps=100,
        save_summary_steps=100,
        save_checkpoints_secs=None,
        save_checkpoints_steps=500,
        keep_checkpoint_max=1
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./model_ckpt/", params=model_params, config=config)

    # ==========  执行任务  ========== #
    if task_type == "train":
        # early_stop_hook 是控制模型早停的控件，下面两个分别是 tf 1.x 和 tf 2.x 的写法
        # early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, metric_name="accuracy",
        early_stop_hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, metric_name="accuracy", max_steps_without_increase=1000, min_steps=500)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=10, batch_size=32), hooks=[early_stop_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=32), steps=None, start_delay_secs=1000, throttle_secs=1)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif task_type == "eval":
        estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=32))
    elif task_type == "infer":
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=32), predict_keys="prob")
        with open("./pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (np.argmax(prob['prob'])))
    if task_type == "export":
        feature_spec = {
            "image": tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name="image"),
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        tf.estimator.Estimator.export_savedmodel("./saved_model/", serving_input_receiver_fn)


if __name__ == '__main__':
    main()