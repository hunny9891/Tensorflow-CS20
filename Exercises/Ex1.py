# Filename: Ex1
# ------------------------------------------------------
# This file is the first file for learning tensorflow.abs

import tensorflow as tf

a = tf.constant(3, name="A")
b = tf.constant(5, name="B")
c = tf.add(a,b, name="ADD")

with tf.Session() as sess:
    result = sess.run(c)

print("The result of first computation i.e a + b is" + str(result))


# Putting a graph on a specific cpu or gpu

# Creates a graph
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], name = 'a')
    b = tf.constant([5.0, 6.0, 7.0], name = 'b')
    c = tf.add(a, b)

# Creates a session with log_device_placement set to True
with tf.Session(config=tf.ConfigProto(log_device_placement = True)) as sess:
    print('This one is from GPU -- ' + str(sess.run(c)))


# Create a graph -- Research more on the commented part below.
g = tf.Graph()

# with g.as_default():
#     x = tf.add(3, 5)

sess = tf.Session(graph = g)
# with tf.Session() as sess:
#     sess.run(x)