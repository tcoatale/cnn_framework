import tensorflow as tf

#%%
def weight_variable(shape, stddev, wd, name):
  with tf.device('/cpu:0'):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial)
  weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss_' + name)
  tf.add_to_collection('losses', weight_decay)
  return var

def bias_variable(shape, constant):
  with tf.device('/cpu:0'):
    initial = tf.constant(constant, shape=shape)
  return tf.Variable(initial) 
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def normalize(input):
  return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def pool_layer(input, size, name):
    return tf.nn.max_pool(input, ksize=[1, size+1, size+1, 1], strides=[1, size, size, 1], padding='SAME', name=name)

def conv2d_layer(input, filter_shape, channels, name):
  shape = filter_shape + [input.get_shape()[3].value] + [channels]
  W = weight_variable(shape, stddev=5e-2, wd=1e-7, name=name)
  b = bias_variable([shape[3]], 0.1)
  return tf.nn.relu(conv2d(input, W) + b, name=name)
  
def fc_layer(input, units, name):
  shape = [input.get_shape()[1].value, units]
  W = weight_variable(shape, stddev=4e-2, wd=4e-3, name=name)
  b = bias_variable([units], 0.1)
  return tf.nn.relu(tf.matmul(input, W) + b, name=name)
  
def readout_layer(input, classes, name):
  shape = [input.get_shape()[1].value, classes]
  W = weight_variable(shape, stddev=1.0 / input.get_shape()[1].value, wd=0.0, name=name)
  b = bias_variable([classes], 0.0)
  return tf.add(tf.matmul(input, W), b, name=name)

def flat(input):
  return tf.reshape(input, [input.get_shape()[0].value, -1])

def softmax_layer(input):
  return tf.nn.softmax(input)
  


