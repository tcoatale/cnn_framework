import tensorflow as tf
import re
import numpy as np
TOWER_NAME = 'tower'


#%%
def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def normalize(input, name):
  return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def conv2d_3x3_raw(input, channels, name):
  shape = [3, 3] + [input.get_shape()[3]] + [channels]
  
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights', shape=shape, stddev=1e-4, wd=0.0)
    biases = _variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
    convolution = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    biased_convolution = tf.nn.bias_add(convolution, biases, name=scope.name)
    normalized = normalize(input=biased_convolution, name=name + '_norm')
  return normalized

def conv2d(filter_shape, channels, input, name, stride=1):
  shape = filter_shape + [input.get_shape()[3]] + [channels]
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights', shape=shape, stddev=1e-4, wd=0.0)
    convolution = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
    biased_convolution = tf.nn.bias_add(convolution, biases)
    biased_nonlinear_convolution = normalize(tf.nn.relu(biased_convolution, name=scope.name), name=scope.name + '_norm')
    
  return biased_nonlinear_convolution
  
def local_layer(units, input, name):
  with tf.variable_scope(name) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = input.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, units], stddev=np.sqrt(2.0/dim), wd=0.004)
    biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
    _activation_summary(local4)
    
  return local4
  
def softmax_layer(units, input, name) :
  with tf.variable_scope(name) as scope:
    dim = input.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', [dim, units], stddev=np.sqrt(2.0/dim), wd=0.0)
    biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.0))
    softmax_linear = tf.nn.softmax(tf.add(tf.matmul(input, weights), biases, name=scope.name))
    _activation_summary(softmax_linear)
    
  return softmax_linear
  
def res_block(input, name):
  channels = input.get_shape()[3].value
  conv1 = conv2d_3x3_raw(input, channels, name=name+'_conv1')
  conv1_relu = tf.nn.relu(conv1, name=name+'_conv1_relu')
  
  conv2 = conv2d_3x3_raw(conv1_relu, channels, name=name+'_conv2')
  res = tf.add(input, conv2, name=name+'_res')
  res_relu = tf.nn.relu(res, name=name+'_res_relu')
  
  return res_relu
  
def inception_res_block(input, name):
  channels = input.get_shape()[3].value

  conv11 = conv2d([1, 1], channels / 2, input, name=name+'_conv11')
  conv12 = conv2d([3, 3], channels / 2, conv11, name=name+'_conv12')

  conv21 = conv2d([1, 1], channels / 2, input, name=name+'conv21')
  conv22 = conv2d([5, 5], channels / 2, conv21, name=name+'_conv22')
  
  concat = tf.concat(3, [conv12, conv22])
  res = tf.add(input, concat, name=name+'_res')
  res_relu = tf.nn.relu(res, name=name+'_res_relu')
  
  return res_relu
  
def red_block(input, name):
  channels = input.get_shape()[3].value
  pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_pool')
  conv = conv2d([1, 1], 2 * channels, pool, name=name+'_conv')
  return conv

def average_pool_output(conv_shape, classes, input, name):
  class_layer = conv2d(conv_shape, classes, input, name + '_conv')

  width = input.get_shape()[1].value
  height = input.get_shape()[2].value
  
  pool_layer = tf.nn.avg_pool(class_layer, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='VALID', name=name + '_pool')  
  reshape = tf.reshape(pool_layer, [-1, classes])
  output = softmax_layer(classes, reshape, name + '_softmax')
  return output