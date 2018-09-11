import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers, seq2seq

tf.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']    #0:h, 1:i, 2:e, 3:l, 4:o
x_data = [[0, 1, 0, 2, 3, 3]]   #hihell
x_one_hot = [[[1,0,0,0,0],  #h 0
              [0,1,0,0,0],  #i 1
              [1,0,0,0,0],  #h 0
              [0,0,1,0,0],  #e 2
              [0,0,0,1,0],  #l 3
              [0,0,0,1,0]]] #l 3

y_data = [[1,0,2,3,3,4]]    #ihello

num_classes = 5
input_dim = 5   #one_hot size
hidden_size = 5 #output from the LSTM. 5 to directly predict ont_hot
batch_size = 1  #one sentence
sequence_length = 6 # |ihello| == 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) #X ont_hot
Y = tf.placeholder(tf.int32, [None, sequence_length])            #Y label

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

#FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
#fc_b = tf.get_variable("fc_b", [num_classes])
#outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

#Reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

#Let's train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction:", result, "true Y:", y_data)

    #Print char using dic
    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("\tPrediction str: ", ''.join(result_str))

# 0 loss: 1.6078763 prediction: [[3 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 1 loss: 1.5102623 prediction: [[3 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 2 loss: 1.4327028 prediction: [[3 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 3 loss: 1.3489527 prediction: [[3 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 4 loss: 1.2551297 prediction: [[1 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 5 loss: 1.140437 prediction: [[1 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
# 6 loss: 1.0167552 prediction: [[1 3 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 7 loss: 0.8969265 prediction: [[1 3 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 8 loss: 0.76952547 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 9 loss: 0.655007 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 10 loss: 0.54275775 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 11 loss: 0.4284713 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 12 loss: 0.33451474 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 13 loss: 0.24750167 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
#                                 .
#                                 .
#                                 .
# 36 loss: 0.0022713589 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 37 loss: 0.002113483 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 38 loss: 0.001977074 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 39 loss: 0.0018580456 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 40 loss: 0.0017534181 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 41 loss: 0.0016607023 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 42 loss: 0.0015781395 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 43 loss: 0.0015044062 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 44 loss: 0.0014382761 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 45 loss: 0.0013789788 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 46 loss: 0.0013257032 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 47 loss: 0.0012777768 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 48 loss: 0.0012345857 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 49 loss: 0.0011956159 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
# 	Prediction str:  ihello
