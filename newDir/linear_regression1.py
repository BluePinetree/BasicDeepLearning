import tensorflow as tf

# #X and Y data
# x_train = [1,2,3]
# y_train = [1,2,3]

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
#-> Now we can use X and Y in place of x_data and y_data
## placeholders for a tensor

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#Our hypothesis XW+b
hypothesis = X*W + b

#cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #Used GradientDescentOptimizer for optimize W,b
train = optimizer.minimize(cost) #minimizes Loss(Cost) value

# Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph
sess.run(tf.global_variables_initializer()) #important!!

#Fit the line
for step in range(2001):
    cost_val, W_val, b_bal, _ = sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_bal)

#Testing our model
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[1.2,5.1]}))

