#Lab 11 MNIST and Deep learning CNN
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)     #reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#hyper parameters
learning_rate = 0.001
trainging_epochs = 15
batch_size = 100

class Model:

    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            #Dropout (keep_prob) rate 0.7~0.5 in traing,
            #but should be 1 in testing
            self.training = tf.placeholder(tf.bool)

            #Input place holders
            self.X = tf.placeholder(tf.float32, [None,784])
            #Img 28x28x1 (black/white). Input layer
            X_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None,10])

            #Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
                # #Strides default value = 1
                # #L1 ImgIn shape=(?, 28, 28, 1)
                # W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
                # L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
                # L1 = tf.nn.relu()

            #Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding='SAME', strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)
                # L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                # L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)


            #Convolutional Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)

            #Pooling Layer #2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], padding='SAME', strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            #Convolutional Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)

            #Pooling Layer #3
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding='SAME', strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            #Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            #Logits (No activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        #Define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.Y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_test, y_test, training=False):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_test, self.Y: y_test, self.training: training})


#Initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print("Learning Started!")

#Train my model
for epoch in range(trainging_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


print('Learning Finished!')

#Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))


# Learning Started!
# Epoch: 0001 cost = 0.160478229
# Epoch: 0002 cost = 0.044477588
# Epoch: 0003 cost = 0.030586271
# Epoch: 0004 cost = 0.022690552
# Epoch: 0005 cost = 0.017413198
# Epoch: 0006 cost = 0.015421592
# Epoch: 0007 cost = 0.012881109
# Epoch: 0008 cost = 0.010547431
# Epoch: 0009 cost = 0.009130055
# Epoch: 0010 cost = 0.008804658
# Epoch: 0011 cost = 0.007421048
# Epoch: 0012 cost = 0.006507725
# Epoch: 0013 cost = 0.006724983
# Epoch: 0014 cost = 0.005328514
# Epoch: 0015 cost = 0.006447479
# Learning Finished!
# Accuracy: 0.9923
