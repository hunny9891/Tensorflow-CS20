# Second exercise on tensorflow learning.
# This focuses on viewing the neural network.

import tensorflow as tf

a = tf.constant(3, name = "a")
b = tf.constant(4, name = "b")
c = tf.add(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))

writer.close()


# Tensors with a specific value
# tf.zeros(shape, dtype=tf.float32, name=None)

zeros = tf.zeros([2,3], dtype=tf.float32, name="b")

# Tensors with a fixed values same as input tensor
# tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# input_tensor  =   [[0, 1], [2, 3], [4, 5]]
input_tensor = [[0, 1], [2, 3], [4, 5]]
alike_tensor = tf.zeros_like(input_tensor)

# same as above works for ones as well
# tf.ones(shape, dtype=tf.float32, name=None)
# tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

# tf.fill(dims, value, name=None)
fillup = tf.fill([2, 3], 8)

# tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
tf.linspace(10.0, 13.0, 4) # ==> [10.0 11.0 12.0 13.0]
# tf.range(start, limit=None, delta=1, dtype=None, name='range')
# 'start' is 3, 'limit' is 18, 'delta' is 3
# tf.range(start, limit, delta) # ==> [3, 6, 9, 12, 15]
# 'limit' is 5 tf.range(limit) ==> [0, 1, 2, 3, 4] 

# Creating randoms

''' tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None) '''

# tf.set_random_seed(seed)

''' a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b) # >> [5 8]
tf.add_n([a, b, b]) # >> [7 10]. Equivalent to a + b + b
tf.mul(a, b) # >> [6 12] because mul is element wise
tf.matmul(a, b) # >> ValueError
tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1])) # >> [[18]]
tf.div(a, b) # >> [1 3] 
tf.mod(a, b) # >> [1 0]
Operations
Pretty standard, quite similar to numpy. See TensorFlow documentation
23 '''

# Only use constants for primitives, otherwise use variables
# Playing with variables

# Create a variable with scalar value
a = tf.Variable(2, name="scalar")

# Create variable b as vector
b = tf.Variable([1, 2], name = "vector")

# Create a 2x2 matrix
c = tf.Variable([[1,0],[9,7]], name = "matrix")

# Sample weight tensor
W = tf.Variable(tf.zeros(784, 10), name = "W")

# Variable should be initialized, therefore do this
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# If required you can initialize a subset of variables also
init_ab = tf.variables_initializer([a, b], name="init_ab") 
with tf.Session() as sess: 
    sess.run(init_ab)

# Intitialize a single variable
VW = tf.Variable(tf.zeros([784,10])) 
with tf.Session() as sess: 
    sess.run(W.initializer)

# To eval a variable just use print var name to print what it really is.
# To see the actual content use var_name.eval()

# On the contarary if we want to assign any value to a variable we need to assign a value
# using assign method i.e var_name.assign() and then run that in session to take effect

# Each session maintains its own copy of variables.

# Interactive session vs Session
sess = tf.InteractiveSession() 
a = tf.constant(5.0) 
b = tf.constant(6.0) 
c = a * b 
# We can just use 'c.eval()' without specifying the context 'sess' 
print(c.eval()) 
sess.close()

# Placeholders -- Why?, Because us or our clients can always pass in data later on.
# create a placeholder of type float 32-bit, shape is a vector of 3 elements 
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements 
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable 
c = a + b  # Short for tf.add(a, b)
with tf.Session() as sess: 
    print(sess.run(c)) # Error because a doesn’t have any value

# create a placeholder of type float 32-bit, shape is a vector of 3 elements 
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements 
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable 
c = a + b  # Short for tf.add(a, b)
with tf.Session() as sess: 
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]} 
    # fetch value of c 
    print(sess.run(c, {a: [1, 2, 3]})) # the tensor a is the key, not the string ‘a’

# Normal loading example
x = tf.Variable(10, name='x') 
y = tf.Variable(20, name='y') 
z = tf.add(x, y) # you create the node for add node before executing the graph
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph) 
    for _ in range(10): 
        sess.run(z) 
    writer.close()

# Lazy loading example
x = tf.Variable(10, name='x') 
y = tf.Variable(20, name='y')
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph) 
    for _ in range(10): 
        sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code 
    writer.close()

# print graph def as a json in command line as well
# tf.get_default_graph().as_graph_def()





