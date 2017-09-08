"""
    This demo uses more TensorFlow function to implement neural network
    Since there're too many new functions for me, this demo is almost a copy other than the comments I added

    Reference:
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network.py

    Somethings learned:
        [1]. tf.estimator is an advanced template. There're 3 main tools: 'train', 'predict' and 'evaluate'
        [2]. In GitHub example, cost function is a cross entropy. Any difference from the one in MatLab?
            --Actually the same!
        [3]. What is tensor layers ?
            --It contains many different kinds of layer. Not very clear yet.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
# here cannot load the data ... not solve yet

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Neural Network parameters:
n_hidden_1 = 256     # In ex4 there're 25 neurons in layer 2
n_hidden_2 = 256
num_input = 784     # Each input x is a 400 dim vector
num_classes = 10    # same as num_label


# Define neural network
def neural_net(x_dict):
    # The estimator input is a dict
    x = x_dict['image']
    # Hidden full-connect layer 1 with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden full-connect layer 2 with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output full-connect layer with neurons for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

def model_fn(features, label, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(label, dtype=tf.int32)
    ))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # evaluate
    acc_op = tf.metrics.accuracy(label, pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)
# Train the model
model.train(input_fn=input_fn, steps=num_steps)

# Evaluate the model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=False
)

# Use estimator evaluate method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
