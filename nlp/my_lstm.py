import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

data = input_data.read_data_sets('/Users/ginray/PycharmProjects/tensorflow_practice/MNIST_data', one_hot=True)

n_inputs = 28
n_steps = 28
hidden_units = 64
batch_size = 100
n_classes = 10
lr = 0.001
training_iters = 100000

xs = tf.placeholder(tf.float32, [None, n_inputs, n_steps])
ys = tf.placeholder(tf.float32, [None, 10])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, hidden_units])),
    'out': tf.Variable(tf.random_normal([hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X):
    X = tf.reshape(X, shape=[-1, n_inputs])
    weight = tf.Variable(tf.random_normal([n_inputs, hidden_units]))  # 注意一定要加 Variable  不然不是可以更新的参数
    bias = tf.Variable(tf.constant(0.1, shape=[hidden_units, ]))

    X = tf.matmul(X, weight) + bias
    X = tf.reshape(X, shape=[-1, n_steps, hidden_units])  # n_steps = 列 ，每一行作为一个step

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(5)])

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # time_major指时间点是不是在主要的维度，因为我们的num_steps在次维，所以定义为了false
    outputs, final_states = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)

    weight = tf.Variable(tf.random_normal([hidden_units, n_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

    # results = tf.matmul(outputs[:, -1, :], weight) + bias
    results = tf.matmul(final_states[-1].c, weight) + bias

    return results


prediction = RNN(xs)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(ys, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        # 一个step是一行
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys}))

        step = step + 1
