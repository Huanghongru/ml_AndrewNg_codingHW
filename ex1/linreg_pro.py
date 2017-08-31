"""
    Since the former one is some what hard-core coding and lack of TensorFlow style
    I mainly refer to the following website:
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
    This is another implementation with more TensorFlow style

    But in TensorFlow, I find it hard to vectorization sometimes...
    Maybe it is just because I'm still not familiar with TensorFLow
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

# Parameters
alpha = 0.01    # learning rate
iters = 1500    # training epoch
display_step = 50   # display some necessary info at each step

# Training data
def load_data(data):
    profit = []
    population = []
    with open(data, 'r') as df:
        for line in df.readlines():
            datas = line.split(',')
            profit.append(eval(datas[0]))
            population.append(eval(datas[1]))
    profit = np.asarray(profit)
    population = np.asarray(population)
    return profit, population

train_X, train_y = load_data("ex1data1.txt")
m = train_X.shape[0]    # num of training datas

# tf Graph input
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

# Set model weights
# In this demo there're only 1 feature so Weight and bias are the same things as theta0 and theta1
theta0 = tf.Variable(rng.randn(), name="theta0")
theta1 = tf.Variable(rng.randn(), name="theta1")

# Construct a linear model
pred = tf.add(tf.multiply(X, theta1), theta0)

# Mean square error
cost = tf.reduce_sum(tf.pow(pred-Y, 2)/(2*m))   # It is called J in MatLab version

# Gradient descent
# NOTE: minimize() knows to modify theta0 and theta1 because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# Initialize the variable (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run initializer firstly
    sess.run(init)

    # Fit all training data:
    for epoch in range(iters):
        for (x, y) in zip(train_X, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(theta1), "b=", sess.run(theta0))

    # The following are some fancy jobs to see your training result
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(theta1), "b=", sess.run(theta0), '\n')

    # Graphic display
    plt.plot(train_X, train_y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(theta1) * train_X + sess.run(theta0), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(theta1) * train_X + sess.run(theta0), label='Fitted line')
    plt.legend()
    plt.show()


