import tensorflow as tf
from configurations.models.blocks.layers import conv2d_layer, pool_layer

def resnet_starter(input, channels):
  with tf.variable_scope('starter'):
    conv = conv2d_layer(input, [11, 11], channels, name='conv')
  return conv

def resnet_macro_block(input, channels, name):
  with tf.variable_scope(name + '_macro'):
    b0 = reduction_block(input, channels, name='b0')
    b1 = resnet_block(b0, name='b1')
    b2 = resnet_block(b1, name='b2')
    b3 = resnet_block(b2, name='b3')
    b4 = resnet_block(b3, name='b4')
  return b4

def resnet_block(input, name):
  channels = input.get_shape()[3].value
  with tf.variable_scope(name + '_residual_block'):
    conv1 = conv2d_layer(input, [1, 1], int(channels / 2), name='conv1')    
    conv2 = conv2d_layer(conv1, [3, 3], channels, name='conv2')
    residual_layer = tf.nn.relu(tf.add(input, conv2), name='output')    
  return residual_layer
  
  
def resnet_inception_block(input, name):
  channels = input.get_shape()[3].value
  with tf.variable_scope(name + '_residual_inception_block'):
    conv_a_1 = conv2d_layer(input, [1, 1], int(channels / 2), name='conv_a_1')
    conv_b_1 = conv2d_layer(input, [1, 1], int(channels / 2), name='conv_b_1')
    
    conv_a_2 = conv2d_layer(conv_a_1, [3, 3], channels, name='conv_a_2')
    conv_b_2 = conv2d_layer(conv_b_1, [5, 5], channels, name='conv_b_2')
    
    concatenation = tf.concat(3, [conv_a_2, conv_b_2], name='concat')
    conv_3 = conv2d_layer(concatenation, [1, 1], channels, name='conv_3')
    
    residual_layer = tf.nn.relu(tf.add(input, conv_3), name='output')    
  return residual_layer
  
def reduction_block(input, channels, name):
  with tf.variable_scope(name + '_reduction_block'):
    conv = conv2d_layer(input, [3, 3], channels, name='conv')
    pool = pool_layer(conv, 2, name='pool')
  return pool



  


    
  