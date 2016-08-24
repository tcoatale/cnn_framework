# -*- coding: utf-8 -*-
from configurations.models.blocks.output_blocks import fc_output, fc_stream
from configurations.models.blocks.layers import conv2d_layer, pool_layer
import tensorflow as tf

#%%
def architecture(input, add_filters):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
  augmented_image = tf.concat(3, [conv1, add_filters])  
  pool2 = pool_layer(augmented_image, 2, name='pool2')
  conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
  pool4 = pool_layer(conv3, 2, name='pool4')
  conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
  dropout_layer = tf.nn.dropout(conv5, 0.75)
  conv6 = conv2d_layer(dropout_layer, [3, 3], 256, name='conv6')
  conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
  pool8 = pool_layer(conv7, 2, name='pool8')
  print(pool8.get_shape())
  return pool8
  

def output(input, features, dataset):
  with tf.variable_scope("feat_augmentation"):
    linear_features = fc_stream(input, [256], name='features')
    all_features = tf.concat(1, [linear_features, features])
  out = fc_output(all_features, dataset.classes, [192, 128], name='output')
  return out
