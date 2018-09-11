# Lab 11 MNIST and Deep learning CNN

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import layers           #To use xavier_initializer()
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) #Reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Hyper parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            #Drop out(keep_prob) rate 0.7~0.5 on training, but should be 1.
            #for training
            self.keep_prob = tf.placeholder(tf.float32)

            #Input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            #Img 28x28x1(black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None,10])

            #L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
            #   Conv    -> (?, 28, 28, 32)
            #   Pool    -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            #L2 ImgIn shape=(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            #   Conv    -> (?, 14, 14, 64)
            #   Pool    -> (?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            #L3 ImgIn shape=(?, 7, 7, 128)
            W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
            #   Conv    -> (?, 7, 7, 128)
            #   Pool    -> (?, 4, 4, 128)
            #   Reshape -> (?, 128 * 4 * 4) for FC(Fully Connected)
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

            #L4 FC 4*4*128 inputs -> 628 outputs
            W4 = tf.get_variable("W4", [128 * 4 * 4, 625], initializer=layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            #L5 FC 625 inputs -> 10 outputs
            W5 = tf.get_variable("W5", [625, 10], initializer=layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4, W5) + b5


        #Define cost/optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_train, y_train, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.keep_prob: keep_prop})



#Initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning started!')

#Train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)         #keep_prop=0.7
        avg_cost += c / total_batch


    print('Epoch:', '%04d' %(epoch + 1), 'Cost= ', '{:.9f}'.format(avg_cost))


print('Learning finished!')

#Test and Check accuracy
print('Accuracy: ', m1.get_accuracy(mnist.test.images, mnist.test.labels))
