"""
    This is a demo that implement ex3 part 1 in TensorFlow.
    'Fmincg' is a function provided by Andrew Ng's course. I can't find a
    similar optimizer in TensorFlow. So the training speed is very low.
    To do a trade off I only train 1000 times.
    The final accuracy is 87.38%.

    Somethings learned:
        [1].mat file can be loaded by scipy.io

"""

import tensorflow as tf
import numpy as np
import scipy.io as scpio


def load_training_data(data_file):
    mat = scpio.loadmat(data_file)
    y, X, _, _, _ = mat.values()
    y.astype(dtype=np.float32)
    X.astype(dtype=np.float32)
    return y, X

train_y, train_X = load_training_data('ex3data1.mat')

# Parameters
report_step = 1000
num_labels = 10
epoches = 1000
_lambda = 0.1
m, n = train_X.shape
train_X = np.concatenate([np.ones([m, 1], dtype=np.float32), train_X], 1)

# Not clear about the element in X, it isn't the gray scale
# Maybe some transition is needed, so I skip the visualization here

X = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
all_theta = tf.Variable(tf.zeros([num_labels, n+1], dtype=tf.float32))


def lr_cost_function(theta, X, y, _lambda):
    """
    In MatLab:
        h = sigmoid(X*theta);
        theta_reg = theta.^2;
        J = (-1/m) * (y'*log(h) + (1-y)'*log(1-h)) + (lambda/(2*m))*sum(theta_reg(2:size(theta)));
        grad = (1/m)*(X'*(h-y)) + (lambda/m)*[0;theta(2:size(theta))];
    :param theta: [(n+1) x 1]
    :param X: [m x (n+1)]
    :param y:
    :param _lambda:
    :return: cost function
    """
    h = tf.sigmoid(tf.matmul(X, theta))
    theta_reg = tf.pow(theta, 2)
    cost = (-1.0/m) * (tf.matmul(tf.transpose(y), tf.log(h)) +
                       tf.matmul(tf.transpose(1-y), tf.log(1-h))) + (_lambda/(2*m))*tf.reduce_sum(theta_reg[1:])
    return cost


def one_vs_all():
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_labels):
            print "Training {0}-th classifier".format(i)

            cur_train_y = tf.cast(tf.equal(y, i+1), dtype=tf.float32)

            cost = lr_cost_function(tf.transpose(tf.reshape(all_theta[i, :], (1, n+1))), X, cur_train_y, _lambda)
            # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
            optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

            for epoch in range(epoches):
                sess.run(optimizer, feed_dict={X: train_X, y: train_y})

                if (epoch+1) % report_step == 0:
                    print "Training step: {0}\t||\tcost: {1}".format(epoch+1,
                                                                     sess.run(cost, feed_dict={X: train_X, y: train_y}))
            print "="*50, "{0}-th classifier training dnoe".format(i), "="*50

        tmp_p = tf.matmul(all_theta, tf.transpose(X))
        p = tf.argmax(tmp_p, axis=0)
        p = tf.reshape(p, (m, 1))
        p = tf.cast(p, dtype=tf.float32)
        e = tf.cast(tf.equal(p+1, y), dtype=tf.float32)

        # print sess.run(e, feed_dict={X: train_X, y: train_y})
        print "Accuracy: {0}".format(sess.run(tf.reduce_mean(e), feed_dict={X: train_X, y: train_y})*100)

one_vs_all()
