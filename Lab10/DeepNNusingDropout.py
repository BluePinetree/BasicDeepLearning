import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
learning_rate = 0.01

#28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

#dropout(keep_prob) rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

#5 layer!
W1 = tf.get_variable('W1', [784,512])
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable('W2', [512,512])
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable('W3', [512, 512])
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', [512, 512])
b4 = tf.Variable(tf.random_normal([512]), name='bias4')
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', [512, nb_classes])
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias5')
hypothesis = tf.matmul(L4, W5) + b5

#Cost(xivier initialization)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Hyper parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    #Initialize tensorflow variables
    sess.run(tf.global_variables_initializer())
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.9})
            avg_cost += c / total_batch

        print('Epoch: ', ' %04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

    #Test model using dataset
    print('Learing Finished!\nAccuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
    # #Get one and predict
    # r = random.randint(0, mnist.test.num_examples - 1)
    # print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X :mnist.test.images[r:r+1]}))
    # plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    # plt.show()
