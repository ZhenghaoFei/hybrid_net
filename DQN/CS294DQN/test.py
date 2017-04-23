import tensorflow as tf



m = tf.Variable([[2, 3], [1, 2], [2, 2], [4, 3]])
# m = [['a', 'b'], ['c', 'd']]
idx = tf.range(0, 32)
a = tf.range(0, 32)
act_idx = tf.stack([idx, a], axis=1)

k = act_idx


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print sess.run(k) 