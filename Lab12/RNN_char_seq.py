#Lab 12 Charactor Sequence RNN

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn, layers, seq2seq

tf.set_random_seed(777) #Reproducibility

sample = " if you want you"
idx2char = list(set(sample))    #index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}   #char -> index(dictionary)
#{'i': 0, 'f': 1, 'w': 2, 'n': 3, 'o': 4, ' ': 5, 'a': 6, 't': 7, 'u': 8, 'y': 9}

#Hyper parameters
dic_size = len(char2idx)    #RNN input size (one hot size)
hidden_size = len(char2idx) #RNN output size
num_classes = len(char2idx) #Fianl output size (RNN or softmax, etc.)
batch_size = 1  #One sample data, one batch
sequence_length = len(sample) - 1   #Number of LSTM rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  #Char to index
#[0, 5, 8, 0, 9, 6, 7, 0, 4, 1, 3, 2, 0, 9, 6, 7]
x_data = [sample_idx[:-1]]  #X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   #Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])   #X data
Y = tf.placeholder(tf.int32, [None, sequence_length])   #Y label

x_one_hot = tf.one_hot(X, num_classes)  #One hot: 1 -> 0 1 0 0 0 0 0 0 0 0
cell =  rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
#Tensor("rnn/transpose_1:0", shape=(1, 15, 10), dtype=float32)

#FC layer
x_for_fc = tf.reshape(outputs, [-1,hidden_size])
outputs = layers.fully_connected(inputs=x_for_fc, num_outputs=num_classes, activation_fn=None)

#Reshape out for sequence loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        #Print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "prediction:", ''.join(result_str))




#                     .
#                     .
#                     .
# 3 loss: 1.8205687 prediction: y   ou   nt you
# 4 loss: 1.5735345 prediction: y  you  nnt you
# 5 loss: 1.2929734 prediction: y  you  ant you
# 6 loss: 1.0188993 prediction: yf you want you
# 7 loss: 0.7543596 prediction: yf you want you
# 8 loss: 0.52488744 prediction: yf you want you
# 9 loss: 0.35725367 prediction: if you want you
# 10 loss: 0.2306659 prediction: if you want you
# 11 loss: 0.14683735 prediction: if you want you
# 12 loss: 0.096990794 prediction: if you want you
# 13 loss: 0.0622696 prediction: if you want you
# 14 loss: 0.04023643 prediction: if you want you
# 15 loss: 0.02745265 prediction: if you want you
# 16 loss: 0.019442817 prediction: if you want you
# 17 loss: 0.014069536 prediction: if you want you
# 18 loss: 0.010388194 prediction: if you want you
# 19 loss: 0.007838917 prediction: if you want you
# 20 loss: 0.0060480307 prediction: if you want you
# 21 loss: 0.004769129 prediction: if you want you
# 22 loss: 0.003840317 prediction: if you want you
# 23 loss: 0.0031542277 prediction: if you want you
# 24 loss: 0.0026388792 prediction: if you want you
# 25 loss: 0.002245166 prediction: if you want you
# 26 loss: 0.0019392611 prediction: if you want you
#                     .
#                     .
#                     .
