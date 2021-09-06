import tensorflow as tf

# with tf.compat.v1.Session() as sess:
#     frase = tf.constant('Olá Mundo!')
#     rodar = sess.run(frase)
#     print(rodar.decode('UTF-8'))

# frase = tf.constant("Olá Mundo!")
# print(frase)

a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d,e)

saida = tf.get_static_value(a)
tf.print(b)
tf.print(c)
tf.print(d)
tf.print(e)
tf.print(f)

print("###################")
print(f)