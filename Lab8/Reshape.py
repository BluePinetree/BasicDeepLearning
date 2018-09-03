import tensorflow as tf
import numpy as np

t = np.array([[[0,1,2],
               [3,4,5]],

              [[6,7,8],
               [9,10,11]]])

with tf.Session():
    print(t.shape)
    print(tf.reshape(t, shape=[-1, 3]).eval()) #(2,2,3) -> (4,3)
    print(tf.reshape(t, shape=[-1, 1, 3]).eval()) #(4,3) -> (4,1,3)

    print(tf.squeeze([[0.], [1.], [2.]]).eval())
    print(tf.expand_dims([0., 1., 2.], 1).eval())
