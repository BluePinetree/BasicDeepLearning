import tensorflow as tf

##Basic
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0) #also tf.float32 implicitly
# node3 = tf.add(node1, node2)
#
# sess = tf.Session()
# print("sess.run(node1,node2): ",sess.run([node1,node2]))
# print("sess.run(node3): ",sess.run(node3))

#PlaceHolder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
sess = tf.Session()
adder_node = a + b # + provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))
