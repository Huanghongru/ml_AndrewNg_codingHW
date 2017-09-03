"""
    This demo is a TensorFlow implementation of ex2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
_lambda = 1     # The regulation parameter
display_step = 50
epochs = 6000

# Training data
def load_data(data_file):
    X = []
    y = []
    with open(data_file, 'r') as df:
        for line in df.readlines():
            datas = line.split(',')
            X.append([eval(datas[0]), eval(datas[1])])
            y.append([eval(datas[2])])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def mapFeature(X):
    # In this problem, X has 2 features
    # We want to set degree to 6
    # So the hypothesis is a polynomial:
    #   X1, X2, X1.^2, X2.^2, X1.*X2.,...
    out = []
    degree = 6
    M, N = X.shape
    for m in range(M):
        out.append([])
        for i in range(1, degree+1):
            for j in range(i+1):
                out[m].append(X[m][0]**(i-j)*X[m][1]**j)
    out = np.array(out)
    return out


def predict(X, theta):
    return tf.cast(tf.sigmoid(tf.matmul(X, theta)) >= 0.5, tf.float32)


def plot_data(train_X, train_y):
    pos = np.equal(train_y, np.ones([train_y.shape[0], 1]).astype(np.float32)).astype(np.float32)
    neg = np.equal(train_y, np.zeros([train_y.shape[0], 1]).astype(np.float32)).astype(np.float32)

    plt.scatter(np.multiply(np.transpose(np.array([train_X[:, 0]])), pos),
                np.multiply(np.transpose(np.array([train_X[:, 1]])), pos), color='red')
    plt.scatter(np.multiply(np.transpose(np.array([train_X[:, 0]])), neg),
                np.multiply(np.transpose(np.array([train_X[:, 1]])), neg), color='green')

train_X, train_y = load_data("ex2data2.txt")
plot_data(train_X, train_y)
plt.show()

train_X = mapFeature(train_X)
m, n = train_X.shape    # num of training datas and features of X
train_X = np.concatenate((np.ones([m, 1]), train_X), 1)

# tf Graph input
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
theta = tf.Variable(tf.zeros([n+1, 1]))
h = tf.sigmoid(tf.matmul(X, theta))

# Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# It must be 1.0/m not 1/m!! Since m is a integer, 1/m will give us 0, then the result is zero
cost = -(1.0/m) * (tf.matmul(tf.transpose(Y), tf.log(h)) +
                 tf.matmul(tf.transpose(tf.ones([m, 1])-Y), tf.log(tf.ones([m, 1])-h)))\
                + (_lambda/(2*m)) * tf.reduce_sum(tf.pow(theta[1:], 2))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

p = predict(X, theta)
accuracy = tf.reduce_mean(tf.cast(tf.equal(p, Y), tf.float32))

# Initialize the variable (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run initialize firstly
    sess.run(init)

    # Fit all training data:
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_y})

        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_y})
            print "Epoch:{0}\tcost={1:.9}".format(epoch+1, c)

    print sess.run(accuracy, feed_dict={X: train_X, Y: train_y})

    u = np.array(np.linspace(-1, 1.5))    # default to 50 steps
    v = np.array(np.linspace(-1, 1.5))
    z = np.zeros([50, 50])
    for i in range(50):
        for j in range(50):
            tmp = np.concatenate([[[1]], mapFeature(np.array([[u[i], v[j]]]))], axis=1)
            z[i, j] = np.matmul(tmp, sess.run(theta))

    z = np.transpose(z)
    plot_data(train_X[:, 1:], train_y)
    plt.contour(u, v, z, [0], color='blue')
    plt.show()