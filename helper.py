import tensorflow as tf
import re
import numpy as np

TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  
def conv_maxpool_norm(shape, stride, input, name):
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights', shape=shape, stddev=1e-4, wd=0.0)
    conv2d = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv2d, biases)
    conv_full = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv_full)
    
  pool = tf.nn.max_pool(conv_full, ksize=[1, stride + 1, stride + 1, 1], strides=[1, stride, stride, 1], padding='SAME', name=name + '_pool')
  norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name + '_norm')
  
  return norm
  
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
