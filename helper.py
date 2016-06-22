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
    
#%%
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


#%%
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

#%%
def normalize(input, name):
  return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def conv2d(filter_shape, channels, input, name, stride=1):
  shape = filter_shape + [input.get_shape()[3]] + [channels]
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights', shape=shape, stddev=1e-4, wd=0.0)
    convolution = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
    biased_convolution = tf.nn.bias_add(convolution, biases)
    biased_nonlinear_convolution = tf.nn.relu(biased_convolution, name=scope.name)
    
  return biased_nonlinear_convolution

def residual_inception(input, name):
  channels = input.get_shape()[3].value
  reduction1 = conv2d([1, 1], channels / 2, input, name + '_reduction1')
  conv11 = conv2d([3, 3], channels, reduction1, name + '_conv11')
  conv12 = conv2d([1, 1], channels / 2, conv11, name + '_conv12')
  conv13 = conv2d([3, 3], 2 * channels, conv12, name + '_conv13')
  
  reduction2 = conv2d([1, 1], channels / 2, input, name + '_reduction2')
  conv21 = conv2d([5, 5], channels, reduction2, name + '_conv21')
  conv22 = conv2d([1, 1], channels / 2, conv21, name + 'conv22')
  conv23 = conv2d([3, 3], 2 * channels, conv22, name + '_conv22')
  
  inception_module = tf.concat(3, [conv13, conv23], name + '_inception')
  inception_reduction = conv2d([1, 1], channels, inception_module, name + '_inception_reduction')
  
  residual_normalized = normalize(tf.add(input, inception_reduction, name=name + '_residual'), name=name + '_residual_norm')
  output= tf.nn.relu(residual_normalized, name=name)
  
  return output
  
def reduction(input, name):
  channels = input.get_shape()[3].value
  
  conv0 = conv2d([1, 1], 2 * channels, input, name + '_conv0')
  
  conv_shield_1 = conv2d([1, 1], channels / 2, conv0, name + '_conv_shield_1')
  conv1 = conv2d([3, 3], channels, conv_shield_1, name + '_conv1', stride=2)
  
  conv_shield_2 = conv2d([1, 1], channels / 2, conv0, name + '_conv_shield_2')  
  conv2 = conv2d([5, 5], channels, conv_shield_2, name + '_conv2')
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_pool2')
  
  concat3 = tf.concat(3, [conv1, pool2], name + '_concat3')
  pool_conv0 = tf.nn.avg_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + '_pool0')  

  residual_reduction = normalize(tf.add(pool_conv0, concat3, name=name + '_residual'), name=name + '_residual_norm')
  output= tf.nn.relu(residual_reduction, name=name)
  return output
  
def conv2d_11(filter_shape, channels, input, name):
  conv11 = conv2d([1, 1], channels, input, name + '_11')
  conv = conv2d(filter_shape, channels, conv11, name)
  return conv
  
def conv_maxpool_norm(filter_shape, channels, stride, input, name):
  biased_nonlinear_convolution = conv2d(filter_shape, channels, input, name)
  _activation_summary(biased_nonlinear_convolution)
  pool = tf.nn.max_pool(biased_nonlinear_convolution, ksize=[1, stride + 1, stride + 1, 1], strides=[1, stride, stride, 1], padding='SAME', name=name + '_pool')
  normalized = normalize(input=pool, name=name + '_norm')
  return normalized

def inception(shapes, channels, stride, input, name):
  with tf.variable_scope(name) as scope:
    convolutions = list(map(lambda shape: conv2d_11(shape, channels, input, scope.name + str(shape[0])), shapes))
    inception_module = tf.concat(3, convolutions, name=scope.name)
    _activation_summary(inception_module)
  pool = tf.nn.max_pool(inception_module, ksize=[1, stride + 1, stride + 1, 1], strides=[1, stride, stride, 1], padding='SAME', name=name + '_pool')
  normalized = normalize(input=pool, name=name + '_norm')
  return normalized
  
#%%
def average_pool_output(conv_shape, classes, input, name):
  class_layer = conv2d(conv_shape, classes, input, name + '_conv')

  width = input.get_shape()[1].value
  height = input.get_shape()[2].value
    
  pool_layer = tf.nn.avg_pool(class_layer, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='VALID', name=name + '_pool')  
  reshape = tf.reshape(pool_layer, [-1, classes])
  output = softmax_layer(classes, reshape, name + '_softmax')
  return output
    
#%%
def local_layer(units, input, name):
  with tf.variable_scope(name) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = input.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, units], stddev=np.sqrt(2.0/dim), wd=0.004)
    biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
    _activation_summary(local4)
    
  return local4

#%%
def alexnet(input, keep_prob, batch_size, classes):
    conv1 = conv2d([11, 11], 96, input, 'conv1', stride=4)
    pool2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    conv3 = conv2d([5, 5], 256, pool2, 'conv3')
    pool4 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    conv5 = conv2d([3, 3], 384, pool4, 'conv5')
    conv6 = conv2d([3, 3], 384, conv5, 'conv6')
    conv7 = conv2d([3, 3], 256, conv6, 'conv7')
    pool8 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool8')
  
    dropout_layer = tf.nn.dropout(pool8, keep_prob)
    reshape = tf.reshape(dropout_layer, [batch_size, -1])  
    local9 = local_layer(2048, reshape, 'local9')
    local10 = local_layer(2048, local9, 'local10')
    softmax_linear = softmax_layer(classes, local10, 'softmax_layer')
    
    return softmax_linear

#%%
def softmax_layer(units, input, name) :
  with tf.variable_scope(name) as scope:
    dim = input.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', [dim, units], stddev=np.sqrt(2.0/dim), wd=0.0)
    biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.0))
    softmax_linear = tf.nn.softmax(tf.add(tf.matmul(input, weights), biases, name=scope.name))
    _activation_summary(softmax_linear)
    
  return softmax_linear