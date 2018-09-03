import tensorflow as tf

with tf.Session():
    t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
    print(tf.reshape(t, shape=[-1, 3]).eval())

