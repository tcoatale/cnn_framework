from configurations.models.layers import conv2d_layer, normalize, pool_layer
import tensorflow as tf

def vgg_block(input, shape, channels):
  conv0 = conv2d_layer(input, shape, channels)
  conv1 = conv2d_layer(conv0, shape, channels)
  pool0 = pool_layer(conv1, 2)
  return pool0
  
def res_block(input, shape):
  channels = input.get_shape()[3].value
  conv0 = conv2d_layer(input, shape, channels)
  conv1 = conv2d_layer(conv0, shape, channels)
  residual = normalize(tf.nn.relu(tf.add(input, conv1)))
  return residual
  
def inception_1(input, channels):
  conv0 = conv2d_layer(input, [1, 1], channels / 2)
  return conv0

def inception_3(input, channels):
  conv0 = conv2d_layer(input, [1, 1], channels / 2)
  conv1 = conv2d_layer(conv0, [3, 3], channels / 2)
  return conv1

def inception_5(input, channels):
  conv0 = conv2d_layer(input, [1, 1], channels / 2)
  conv1 = conv2d_layer(conv0, [5, 5], channels / 2)
  return conv1
        
def inc_res_block(input):
  channels = input.get_shape()[3].value
  inc_1 = inception_1(input, channels)
  inc_3 = inception_3(input, channels)
  inc_5 = inception_5(input, channels)
  aggregate = tf.concat(3, [inc_1, inc_3, inc_5])
  conv1 = conv2d_layer(aggregate, [1, 1], channels)
  residual = normalize(tf.nn.relu(tf.add(input, conv1)))
  return residual

def red_block(input):
  channels = input.get_shape()[3].value
  conv1 = conv2d_layer(input, [3, 3], 2 * channels)
  pool1 = pool_layer(conv1, 2)
  return pool1
    
 
''' 
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
'''