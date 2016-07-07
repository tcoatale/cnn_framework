import tensorflow as tf

#%%
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  var = tf.Variable(initial)
  weight_decay = tf.mul(tf.nn.l2_loss(var), 1e-4, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)
  return var

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def normalize(input):
  return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def conv2d_layer(input, filter_shape, channels):
  shape = filter_shape + [input.get_shape()[3].value] + [channels]
  W = weight_variable(shape)
  b = bias_variable([shape[3]])
  
  return normalize(tf.nn.relu(conv2d(input, W) + b))
  
def fc_layer(input, units):
  shape = [input.get_shape()[1].value, units]
  W = weight_variable(shape)
  b = bias_variable([units])
  
  return tf.nn.relu(tf.matmul(input, W) + b)
  
def readout_layer(input, classes):
  shape = [input.get_shape()[1].value, classes]
  W = weight_variable(shape)
  b = bias_variable([classes])
  
  return tf.matmul(input, W) + b

def softmax_layer(input):
  return tf.nn.softmax(input)

#%%
  
def flat(input):
  return tf.reshape(input, [input.get_shape()[0].value, -1])

def average_pool_vector(conv_shape, outputs, input, name):
  outputs_layer = conv2d(conv_shape, outputs, input, name + '_conv')
  width = input.get_shape()[1].value
  height = input.get_shape()[2].value
  pool_layer = tf.nn.avg_pool(outputs_layer, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='VALID', name=name + '_pool')
  reshape = tf.reshape(pool_layer, [-1, outputs])
  
  return reshape

def average_pool_output(conv_shape, classes, input, name):
  activations = average_pool_vector(conv_shape, classes, input, name)
  output = softmax_layer(classes, activations, name + '_softmax')
  return output
  


