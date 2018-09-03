import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

#28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

#2 layer!
with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([784, 100]), name='weight1')
    b1 = tf.Variable(tf.random_normal([100]), name='bias1')

    W1_hist = tf.summary.histogram('weights1',W1)
    b1_hist = tf.summary.histogram('bias1',b1) #Add histogram on TensorBoard

    layer1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    layer1_hist = tf.summary.histogram('layer1', layer1)

with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([100, nb_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2')

    W2_hist = tf.summary.histogram('weights2',W2)
    b1_hist = tf.summary.histogram('bias2',b2) #Add histogram on TensorBoard

    hypothesis = tf.nn.softmax(tf.matmul(layer1, W2) + b2)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis) #Add histogram on TensorBoard

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

cost_summ = tf.summary.scalar('cost', cost) #Add scolor on TensorBoard

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#Merge all summery
summary = tf.summary.merge_all()

#Hyper parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    #Initialize tensorflow variables
    sess.run(tf.global_variables_initializer())

    #Create summary writer
    writer = tf.summary.FileWriter('./logs/xor_logs')
    writer.add_graph(sess.graph)


    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        #Add summary per epoch!
        s = sess.run(summary, feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(s, global_step=epoch)

        #Print cost value per epoch
        print('Epoch: ', ' %04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

    #Test model using dataset
    print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    #Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X :mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
