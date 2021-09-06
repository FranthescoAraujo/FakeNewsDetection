import tensorflow as tf

A = tf.constant([[1,2,3],[4,5,6]])
B = tf.constant([[0,0],[1,0],[0,1]])

C = tf.matmul(A,B)

tf.print(C)