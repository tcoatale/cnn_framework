import tensorflow as tf
from configurations.models.layers import conv2d_layer

def resnet_inception_block(input, name):
  channels = input.get_shape()[3].value
  with tf.variable_scope(name):
    conv_a_1 = conv2d_layer(input, [1, 1], channels, name='conv_a_1')
    conv_b_1 = conv2d_layer(input, [1, 1], channels, name='conv_b_1')
    
    conv_a_2 = conv2d_layer(conv_a_1, [3, 3], channels, name='conv_a_2')
    conv_b_2 = conv2d_layer(conv_b_1, [5, 5], channels, name='conv_b_2')
    
    concatenation = tf.concat(3, [conv_a_2, conv_b_2], name='concat')
    conv_3 = conv2d_layer(concatenation, [1, 1], channels, name='conv_3')
    
    residual_layer = tf.nn.relu(tf.add(input, conv_3), name='output')    
  return residual_layer
  

    
    
  