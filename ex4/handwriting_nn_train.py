"""
    This demo is a raw implementation of ex4 in TensorFlow
    The neural network in ex4 is a 1-Hidden layer

    Somethings learned:
        [1]. Are 'weight and bias' implementation the same as MatLab implementation ?
        [2]. In GitHub example, cost function is a cross entropy. Any difference from the one in MatLab?
        [3]. What is tensor layers ?
"""

import tensorflow as tf
import numpy as np
import scipy.io as scpio

# parameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
Lambda = 1

# load training data and weights
def load_data(data_file):
    mat = scpio.loadmat(data_file)
    y, X, _, _, _ = mat.values()
    return y, X

def load_weights(data_file):
    mat = scpio.loadmat(data_file)
    theta2, _, _, theta1, _ = mat.values()
    return theta1, theta2

train_y, train_x = load_data("ex4data1.mat")
theta1, theta2 = load_weights("ex4weights.mat")
m, n = train_x.shape
train_x = np.concatenate([np.ones([m, 1]), train_x], 1)
train_y_pro = np.zeros([m, num_labels])
for i in range(m):
    train_y_pro[i, train_y[i]-1] = 1

# Graph Input
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
Theta1 = tf.Variable(tf.zeros(theta1.shape), dtype=tf.float32)
Theta2 = tf.Variable(tf.zeros(theta2.shape), dtype=tf.float32)

def nn_cost_function(Theta1, Theta2,
                     num_labels, X, y, Lambda):
    """
    In MatLab:
        % compute h
        X = [ones(m, 1) X];	% 5000 x 401 also a1
        z2 = X * Theta1';
        a2 = sigmoid(z2)	% 5000 x 25

        z2 = [ones(m, 1) z2];	%5000 x 26
        a2 = [ones(m, 1) a2];
        z3 = a2 * Theta2';	% 5000 x 10
        h = sigmoid(z3);	% 5000 x 10	also a3

        % preprocess y to compute the cost
        y_ = zeros(m, num_labels);	% 5000 x 10
        for i = 1 : m
            y_(i, y(i)) = 1;
        end

        % compute J also you can use trace to fully vectorization
        % but it does too many useless job
        % More elegant way: unroll it to a vector and then do multiply
        y_v = [y_(:)];
        h_ = [h(:)];
        theta1_sq = Theta1.^2;
        theta2_sq = Theta2.^2;
        J = -(1/m) * (y_v'*log(h_) + (1-y_v')*log(1-h_)) + ...
            (lambda/(2*m))*(sum(sum(theta1_sq(:, 2: end))) + ...
                            sum(sum(theta2_sq(:, 2: end))));
    :return: cost
    """
    # feed forward to compute h
    z2 = tf.matmul(X, tf.transpose(Theta1))
    a2 = tf.sigmoid(z2)     # 5000 x 25

    z2 = tf.concat([tf.ones([m, 1]), z2], 1)    # 5000 x 26
    a2 = tf.concat([tf.ones([m, 1]), a2], 1)
    z3 = tf.matmul(a2, tf.transpose(Theta2))
    h = tf.sigmoid(z3)

    y_v = tf.reshape(Y, [-1, 1])
    h_ = tf.reshape(h, [-1, 1])
    theta1_sq = tf.pow(Theta1, 2)
    theta2_sq = tf.pow(Theta2, 2)
    J = -(1.0/m) * (tf.matmul(tf.transpose(y_v), tf.log(h_)) +
                    tf.matmul(1-tf.transpose(y_v), tf.log(1-h_))) + \
        (Lambda/(2.0*m))*(tf.reduce_sum(theta1_sq) + tf.reduce_sum(theta2_sq))
    return J

cost = nn_cost_function(Theta1, Theta2, num_labels, X, Y, Lambda)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print "cost: {0}".format(sess.run(cost, feed_dict={X: train_x, Y: train_y_pro,
                                                       Theta1: theta1, Theta2: theta2}))
    


