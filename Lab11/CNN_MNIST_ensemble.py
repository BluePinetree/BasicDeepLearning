#Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) #Reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:

    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            #Dropout (keep_prob) rate 0.7~0.5 on training
            #but should be 1 for testing
            self.training = tf.placeholder(tf.bool)

            #Input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            #img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            #Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)

            #Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, padding='SAME')
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            #Convolutional Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)

            #Pooling Layer #2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, padding='SAME')
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            #Convolutional Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3] ,padding='SAME', activation=tf.nn.relu)

            #Pooling Layer #3
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2, padding='SAME')
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            #Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            #Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        #Define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_train, y_train, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.training: training})


#Initialize
sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print("Learning Started!")

#Train my model!
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        #Train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print("Learning Finished!")

#Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p


ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

# Learning Started!
# Epoch: 0001 cost = [0.28706979 0.28341451]
# Epoch: 0002 cost = [0.08947633 0.08571514]
# Epoch: 0003 cost = [0.06831952 0.06611165]
# Epoch: 0004 cost = [0.05587902 0.05553539]
# Epoch: 0005 cost = [0.04772123 0.05169922]
# Epoch: 0006 cost = [0.04531981 0.04356397]
# Epoch: 0007 cost = [0.04118702 0.04212548]
# Epoch: 0008 cost = [0.03758763 0.03726894]
# Epoch: 0009 cost = [0.03607628 0.03720126]
# Epoch: 0010 cost = [0.03248821 0.03278537]
# Epoch: 0011 cost = [0.03256225 0.03102869]
# Epoch: 0012 cost = [0.03038242 0.02931954]
# Epoch: 0013 cost = [0.02844371 0.02810837]
# Epoch: 0014 cost = [0.02755092 0.02861199]
# Epoch: 0015 cost = [0.02533321 0.02777721]
# Learning Finished!
# 0 Accuracy: 0.9932
# 1 Accuracy: 0.9943
# Ensemble accuracy: 0.9946
