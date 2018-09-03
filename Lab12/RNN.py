import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
import numpy as np

char_rdic = ['h','e','l','o']    #id -> char
char_dic = {w:i for i, w in enumerate(char_rdic)}    #char -> id
ground_truth = [char_dic[c] for c in "hello"] #to index

x_data = np.array([[1,0,0,0], #'h'
                    [0,1,0,0], #'e'
                    [0,0,1,0], #'l'
                    [0,0,1,0]], #'l'
                  dtype=np.float32)
x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)

#Configuration
rnn_size = len(char_dic)
batch_size = 1      #one sample
output_size = 4

#RNN model
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size, activation=tf.nn.tanh, dtype=tf.float32)
initial_state = rnn_cell.zero_state(batch_size, tf.float32)
initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size])
x_split = tf.split(x_data, len(char_dic), 0)    #가로축 4개로 split

outputs, state = tf.nn.static_rnn(cell = rnn_cell, inputs = x_split, initial_state = initial_state)
logits = tf.reshape(tf.concat(outputs, 1),[-1, rnn_size])         #4 x 4
targets = tf.reshape(ground_truth[1:], [-1])    #a shape of [-1] flattens into 1-D
weights = tf.ones([len(char_dic) * batch_size])

loss = legacy_seq2seq.sequence_loss_by_example([logits],[targets],[weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    #initialize variables
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])
