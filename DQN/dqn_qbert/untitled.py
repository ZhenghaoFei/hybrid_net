import numpy as np
import tensorflow as tf

beta = tf.Variable(0.5*np.ones(10), name='beta', dtype=tf.float32)
sess = tf.Session()
init = tf.global_variables_initializer()

k = beta[1]*2
sess.run(init)
a = sess.run(k)
print(a)