"""
    This is a demo that implement part 2 of ex3
"""

import tensorflow as tf
import numpy as np
import scipy.io as scpio


def load_data(data_file):
    y, X, _, _, _ = scpio.loadmat(data_file).values()
    y.astype(dtype=np.float32)
    X.astype(dtype=np.float32)
    return y, X


def load_weight(data_file):
    theta_2, _, _, theta_1, _ = scpio.loadmat(data_file).values()
    theta_1.astype(dtype=np.float32)
    theta_2.astype(dtype=np.float32)
    return theta_1, theta_2

train_y, train_x = load_data("ex3data1.mat")
theta_1, theta_2 = load_weight("ex3weights.mat")
m, n = train_x.shape
train_x = np.concatenate([np.ones([m, 1], dtype=np.float32), train_x], 1)   # add 1s column

# TensorFlow Variables
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
THETA_1 = tf.placeholder(dtype=tf.float32)
THETA_2 = tf.placeholder(dtype=tf.float32)


def predict(Theta1, Theta2, X, Y):
    """
    In MatLab:
        X = [ones(m, 1) X];
        z2 = Theta1 * X';
        a2 = sigmoid(z2);

        a2 = [ones(1, size(a2, 2)); a2];
        z3 = Theta2 * a2;
        a3 = sigmoid(z3);
        [~, i] = max(a3, [], 1);
        p = i';
    :return: the result of classification
    """
    # layer 2   Theta1: 25 x 401    a2: 25 x 5000
    z2 = tf.matmul(Theta1, tf.transpose(X))
    a2 = tf.sigmoid(z2)

    # layer 3   Theta2: 10 x 26     a3: 10 x 5000
    a2 = tf.concat([tf.ones([1, m]), a2], 0)
    z3 = tf.matmul(Theta2, a2)
    a3 = tf.sigmoid(z3)

    # get prediction result p: 1 x 5000
    p = tf.argmax(a3, axis=0) + 1   # in MatLab index starts at 1
    p = tf.cast(tf.reshape(p, [1, m]), dtype=tf.float32)
    p = tf.transpose(p)

    e = tf.cast(tf.equal(p, Y), dtype=tf.float32)

    return p, tf.reduce_mean(e)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    p, accuracy = predict(THETA_1, THETA_2, X, Y)
    print "Accuracy: {0}".format(100*sess.run(accuracy,
                                              feed_dict={THETA_1: theta_1, THETA_2: theta_2,
                                                         X: train_x, Y: train_y}))
