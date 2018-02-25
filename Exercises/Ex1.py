# Filename: Ex1
# ------------------------------------------------------
# This file is the first file for learning tensorflow.abs

import tensorflow as tf

a = tf.constant(3)
b = tf.constant(5)
c = tf.add(a,b)

with tf.Session() as sess:
    result = sess.run(c)

print(result)