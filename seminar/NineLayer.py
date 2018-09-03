import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

X = tf.placeholder(tf.float32, [None,8])
Y = tf.placeholder(tf.float32, [None,1])

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
x_data = MinMaxScaler(x_data)
y_data = xy[:, [-1]]

#Our hypothesis
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_uniform([8,5], -1.0, 1.0), name='weight1')
    b1 = tf.Variable(tf.zeros([5]), name='bias1')
    L1 = tf.sigmoid(tf.matmul(X,W1) + b1)
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight2')
    b2 = tf.Variable(tf.zeros([5]), name='bias2')
    L2 = tf.sigmoid(tf.matmul(L1,W2) + b2)
with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight3')
    b3 = tf.Variable(tf.zeros([5]), name='bias3')
    L3 = tf.sigmoid(tf.matmul(L2,W3) + b3)
with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight4')
    b4 = tf.Variable(tf.zeros([5]), name='bias4')
    L4 = tf.sigmoid(tf.matmul(L3,W4) + b4)
with tf.name_scope("layer5") as scope:
    W5 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight5')
    b5 = tf.Variable(tf.zeros([5]), name='bias5')
    L5 = tf.sigmoid(tf.matmul(L4,W5) + b5)
with tf.name_scope("layer6") as scope:
    W6 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight6')
    b6 = tf.Variable(tf.zeros([5]), name='bias6')
    L6 = tf.sigmoid(tf.matmul(L5,W6) + b6)
with tf.name_scope("layer7") as scope:
    W7 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight7')
    b7 = tf.Variable(tf.zeros([5]), name='bias7')
    L7 = tf.sigmoid(tf.matmul(L6,W7) + b7)
with tf.name_scope("layer8") as scope:
    W8 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight8')
    b8 = tf.Variable(tf.zeros([5]), name='bias8')
    L8 = tf.sigmoid(tf.matmul(L7,W8) + b8)
with tf.name_scope("layer9") as scope:
    W9 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight9')
    b9 = tf.Variable(tf.zeros([5]), name='bias9')
    L9 = tf.sigmoid(tf.matmul(L8,W9) + b9)
with tf.name_scope("layer10") as scope:
    W10 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='weight10')
    b10 = tf.Variable(tf.zeros([5]), name='bias10')
    L10 = tf.sigmoid(tf.matmul(L9,W10) + b10)
with tf.name_scope("last") as scope:
    W11 = tf.Variable(tf.random_uniform([5,1], -1.0, 1.0), name='weight11')
    b11 = tf.Variable(tf.zeros([1]), name='bias11')
    hypothesis = tf.matmul(L10,W11) + b11


cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
cost_summ = tf.summary.scalar('cost', cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar('accuracy', accuracy)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/DeepSigmoid')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    feed_train = {X: x_data, Y: y_data}
    for step in range(8001):
        s ,h, c, _ = sess.run([summary, hypothesis, cost, train], feed_dict=feed_train)
        if step % 50 == 0:
            print(step, c)
            writer.add_summary(s, global_step=step)

    print("Accuracy: ", sess.run(accuracy, feed_dict=feed_train))
