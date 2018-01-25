#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## 使用 Tensorflow 来建立线性回归模型
## 验证
# - y = Wx + b, 首先构造出1000个点，W= 0.1, b = 0.3
# - 再用 tf 去找出 什么样的 W, b 可以拟合上一步所产生的点


# 1000 points
num_points = 1000
vector_set = []

for _ in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = 0.1 * x1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])


x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]
fig = plt.figure(figsize=(10, 8))
fig.add_subplot(111)
plt.scatter(x_data, y_data, c='r')
plt.show()



# using Tensorflow

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data), name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train = optimizer.minimize(loss, name='train')

print ('W=', W, 'b=', b, 'loss=', loss)

# train 30 steps

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(30):
    sess.run(train)
    print ('Step.%s' % i, 'W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))

plt.scatter(x_data, y_data, c='r')

plt.plot(x_data, sess.run(W)*x_data+sess.run(b), c='b')
plt.show()







