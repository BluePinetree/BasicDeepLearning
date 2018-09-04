#Lab 11 MNIST and Convolutional Neural Network
#Coding Practice
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers

tf.set_random_seed(777) #reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

#Hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])      #img 28 x 28 x 1(black/white)
Y = tf.placeholder(tf.float32, [None, 10])

#L1 ImgIn shape(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
#   Conv ->(?, 28, 28 ,32)
#   Pool ->(?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
# Conv ->(?, 14, 14, 64)
# Pool ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

#Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=layers.xavier_initializer()) #contrib 모듈에서 xavier초기값 가져오기
#tf.get_variable: Gets an existing variable with these parameters or create a new one.
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

#Define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Train My Model
with tf.Session() as sess:
    #Initialize
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes some time.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch: ', '%0.4d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost)) #Cost per each epoch

    print('Learning finished!')

    #Test Model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Learning started. It takes some time.
    # Epoch:  0001 cost= 0.345631706
    # Epoch:  0002 cost= 0.091846461
    # Epoch:  0003 cost= 0.068315066
    # Epoch:  0004 cost= 0.056402526
    # Epoch:  0005 cost= 0.047012293
    # Epoch:  0006 cost= 0.041039093
    # Epoch:  0007 cost= 0.036779450
    # Epoch:  0008 cost= 0.032759834
    # Epoch:  0009 cost= 0.027927846
    # Epoch:  0010 cost= 0.024889312
    # Epoch:  0011 cost= 0.022389571
    # Epoch:  0012 cost= 0.020563705
    # Epoch:  0013 cost= 0.016964667
    # Epoch:  0014 cost= 0.015320329
    # Epoch:  0015 cost= 0.013451788
    # Learning finished!

    #Get one and Predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Lable:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('Prediction:', sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
