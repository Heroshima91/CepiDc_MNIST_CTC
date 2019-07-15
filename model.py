# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



BATCH_SIZE = 100
SEQ_LENGTH = 12
tf.logging.set_verbosity(tf.logging.INFO)

# Create model of CNN with slim api
def inference(inputs, params, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        ):
        x = tf.reshape(inputs, [-1, 28, 100, 1])
        net = slim.conv2d(x, int(params['hidden_size']*(params['progression']**1)), params['kernel_size'], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, int(params['hidden_size']*(params['progression']**2)), params['kernel_size'], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, int(params['hidden_size']*(params['progression']**3)), params['kernel_size'], scope='conv3')
        net = slim.max_pool2d(net, [7, 2], scope='pool3')
        # net = slim.conv2d(net, [7, 3], scope='fully_conv', padding='VALID')
        net = tf.squeeze(net, axis=1)
        net = slim.fully_connected(net, int(params['hidden_size']*(params['progression']**4)), scope='fc1')
        net = slim.dropout(net, params['drop_out'])
        net = slim.fully_connected(net, int(params['hidden_size']*(params['progression']**5)), scope='fc2')
        net = slim.dropout(net, params['drop_out'])
        outputs = slim.fully_connected(net, 11, activation_fn=None, normalizer_fn=None, scope='fco')
        return outputs


def get_loss(logits, sequence, sequence_length):

    # pour time_major=True
    # logits = tf.transpose(logits, [1, 0, 2])
    losses = tf.nn.ctc_loss(labels=sequence,inputs=logits,sequence_length=sequence_length, time_major=True)
    losses = tf.reduce_mean(losses)
    tf.summary.scalar('ctc_loss', losses)
    return losses


def get_train_op(loss,learning_rate):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer.minimize(loss, global_step=global_step)


def model_fn(features, labels, params, mode):
    logits = inference(features, params, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.transpose(logits, [1, 0, 2])
    sequence_length = tf.ones([BATCH_SIZE], dtype=tf.dtypes.int32) * SEQ_LENGTH

    if mode == tf.estimator.ModeKeys.PREDICT:

        # maybe reshape?
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            logits,
            sequence_length,
            top_paths=1,
            beam_width=1,
            merge_repeated=False
        )
        prediction = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        print(prediction)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    else:
        labels = tf.string_split(labels, delimiter="").values
        labels = tf.string_to_number(labels, tf.int32)
        labels = tf.reshape(labels, [BATCH_SIZE, 5])
        zero = tf.constant(-1, dtype=tf.int32)
        idx = tf.where(tf.not_equal(labels, zero))
        sequence = tf.SparseTensor(idx, tf.gather_nd(labels, idx), labels.get_shape())

        loss = get_loss(logits=logits, sequence=sequence, sequence_length=sequence_length)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # get loss and training op
            train_op = get_train_op(loss,params['learning_rate'])    # ajouter params['train'] pour gérer learning rate en exploration hparams
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
                logits,
                sequence_length,
                top_paths=1,
                beam_width=1,
                merge_repeated=False
            )
            cast = tf.cast(decoded[0], tf.int32)
            prediction = tf.sparse_tensor_to_dense(cast, default_value=-1)
            prediction = tf.concat(prediction,axis=1)
            concatit = tf.ones([BATCH_SIZE, 5], tf.int32)*(-1)
            prediction = tf.concat([prediction,concatit],axis=1)
            prediction = prediction[:, :5]
            eval_metrics = {
                'accuracy': tf.metrics.accuracy(
                    tf.sparse_tensor_to_dense(sequence),
                    prediction,
                    name='accuracy'
                )
            }
            return tf.estimator.EstimatorSpec(mode, predictions=prediction, loss=loss, eval_metric_ops=eval_metrics)








"""
prediction = list(estimator.predict(
    input_fn=test_input_fn
))

a = [arr.tolist() for arr in prediction]
b = []
for elt in a:
    c = elt[:4]
    b.append(c)
b = np.array(b)
lab_list = lab_test.tolist()
lab_int = []
for elt in lab_list:
    lab_int.append([int(d) for d in str(elt)])
lab_int = np.array(lab_int)
sec = np.sum(np.all(b == lab_int, axis=1))
number_of_equal_elements = np.sum(b==lab_int)
total_elements = np.multiply(*b.shape)
percentage = number_of_equal_elements/total_elements
print('total number of elements: \t\t{}'.format(total_elements))
print('number of identical elements: \t\t{}'.format(number_of_equal_elements))
print('number of different elements: \t\t{}'.format(total_elements-number_of_equal_elements))
print('percentage of identical elements: \t{:.2f}%'.format(percentage*100))"""

