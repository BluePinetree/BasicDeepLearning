import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(5.0)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Manual gradient
gradient = tf.reduce_mean((W*X - Y)*X) * 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#Get Gradients
gvs = optimizer.compute_gradients(cost)

#Apply Gradients
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
