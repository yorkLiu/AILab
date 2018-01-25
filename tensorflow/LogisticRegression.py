#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import input_data


mnist = input_data.read_data_sets("data/", one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print trainimg.shape
print testimg.shape
print trainlabel.shape
print testlabel.shape
print testlabel[0]



# 构建 Tensorflow
x = tf.placeholder('float', [None, 784], name='x')
y = tf.placeholder('float', [None, 10],name='y')
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the Logistic Regression function (softmax)
logic_softmax = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss function
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(logic_softmax), reduction_indices=1))

# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# Prediction
pred = tf.equal(tf.arg_max(logic_softmax, 1), tf.arg_max(y, 1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(pred, 'float'))

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

training_epochs = 50
batch_size = 100
display_size = 5

for epoch in range(training_epochs):
    avg_loss = 0
    batch_numbers = mnist.train.num_examples/batch_size
    for i in range(batch_numbers):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed = {x: batch_xs, y:batch_ys}
        sess.run(optimizer, feed_dict=feed)

        avg_loss += sess.run(loss, feed_dict=feed)/batch_numbers

    # Display
    if epoch % display_size == 0:
        train_feed = {x:batch_xs, y: batch_ys}
        test_feed = {x: mnist.test.images, y: mnist.test.labels}

        train_accuracy = sess.run(accuracy, feed_dict=train_feed)
        test_accuracy = sess.run(accuracy, feed_dict=test_feed)

        print ("Epoch %3d/%3d Loss: %.9f, train_accuracy: %.3f, test_accuracy: %.3f" %
               (epoch, training_epochs, avg_loss, train_accuracy, test_accuracy))
