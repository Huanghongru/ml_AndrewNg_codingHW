"""
    This demo is a TensorFlow implementation of exercise1 in AndrewNg's course--Machine Learning
"""

import tensorflow as tf


def compute_cost_function(X, y, theta, m):
    X = tf.concat([tf.ones([m, 1]), X], 1)
    h = tf.matmul(X, theta)
    J = (1/2*m) * tf.ones([1, m]) * (tf.pow((h-y), 2*tf.ones([m, 1])))
    return J


def gradient_descent(X, y, theta, alpha, m, num_iters):
    X = tf.concat([tf.ones([m, 1]), X], 1)
    for i in range(num_iters):
        h = tf.matmul(X, theta)
        theta = theta - (alpha/m) * (tf.matmul(tf.transpose(X), h-y))
    return theta


def load_data(data):
    profit = []
    population = []
    with open(data, 'r') as df:
        for line in df.readlines():
            datas = line.split(',')
            profit.append(eval(datas[0]))
            population.append(eval(datas[1]))
    m = len(profit)
    profit = tf.transpose([profit])
    population = tf.transpose([population])
    return profit, population, m

# load data:
data_file = "ex1data1.txt"
profit, population, m = load_data(data_file)

# define the computing graph
# Since we implement it in TensorFlow, we should set X and y as tf.placeholder
# and feed them when call tf.Session().run()
alpha = 0.01
iteration = 1500
X = tf.placeholder(tf.float32, [m, 1])   # X has 1 features in ex1data1
y = tf.placeholder(tf.float32, [m, 1])   #
theta = tf.Variable(tf.zeros([2, 1]))
J = compute_cost_function(X, y, theta, m)
final_theta = gradient_descent(X, y, theta, alpha, m, iteration)

# train_step = tf.train.GradientDescentOptimizer(alpha).minimize(J, theta)

def train():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(final_theta, feed_dict={X: profit.eval(session=sess), y: population.eval(session=sess)})

train()


