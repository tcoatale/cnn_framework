import tensorflow as tf
from configurations.models.variables import weight_variable, bias_variable, conv2d

#%%
def normalize(input):
  return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def pool_layer(input, size, name):
    return tf.nn.max_pool(input, ksize=[1, size+1, size+1, 1], strides=[1, size, size, 1], padding='SAME', name=name)

def conv2d_layer(input, filter_shape, channels, name):
  shape = filter_shape + [input.get_shape()[3].value] + [channels]
  W = weight_variable(shape, stddev=5e-2, wd=1e-7, name=name+'_kernel')
  b = bias_variable([shape[3]], 0.1, name=name+'_bias')
  return tf.nn.relu(conv2d(input, W) + b, name=name)
  
def fc_layer(input, units, name):
  shape = [input.get_shape()[1].value, units]
  W = weight_variable(shape, stddev=4e-2, wd=4e-3, name=name+'_kernel')
  b = bias_variable([units], 0.1, name=name+'_bias')
  return tf.nn.relu(tf.matmul(input, W) + b, name=name)
  
def readout_layer(input, classes, name):
  shape = [input.get_shape()[1].value, classes]
  W = weight_variable(shape, stddev=1.0 / input.get_shape()[1].value, wd=0.0, name=name+'_kernel')
  b = bias_variable([classes], 0.0, name=name+'_bias')
  return tf.add(tf.matmul(input, W), b, name=name)

def flat(input):
  return tf.reshape(input, [input.get_shape()[0].value, -1])

def softmax_layer(input):
  return tf.nn.softmax(input)