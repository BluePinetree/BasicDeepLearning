import numpy as np
import tensorflow as tf

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7 #1~7

X = tf.placeholder(tf.float32, [None,16])
Y = tf.placeholder(tf.int32, [None,1]) #1~7

Y_one_hot = tf.one_hot(Y, nb_classes) #one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1,nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

#tf.nn.softmax computes softmax activations
#softmax = exp(Logits) / reduce_sum(exp(Logits, dim))
logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

#Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(optimizer, feed_dict=feed)
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict=feed)
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p,y in zip(pred, y_data.flatten()):
        print("[{}] Prediction {} True Y: {}".format(p==int(y), p, int(y)))
