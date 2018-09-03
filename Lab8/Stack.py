import tensorflow as tf

with tf.Session():
    x = [1, 4]
    y = [2, 5]
    z = [3, 6]

    #Pack along first dim
    print(tf.stack([x,y,z]).eval())

    print(tf.stack([x,y,z], axis=-1).eval()) #-1은 제일 마지막 축을 잡는다
