import tensorflow as tf


w = tf.Variable([[1, 0]])
x = tf.Variable([[2],[ 0]])
y = tf.matmul(w, x)

print w
print y

init_opt = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_opt)
    print y.eval()

# tf.train.Saver
w = tf.constant([1, 3, 3, 4, 5, 6], shape=(3, 2))
x = tf.constant([5, 6, 7, 8, 9, 1], shape=(2, 3))

y = tf.matmul(w, x)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(y)

    save_path = saver.save(sess, './Test')
    print save_path


