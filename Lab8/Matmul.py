import tensorflow as tf

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
print(tf.Session().run(tf.matmul(matrix1, matrix2)))
print(tf.Session().run(matrix1 * matrix2))
