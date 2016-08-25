# -*- coding: utf-8 -*-
from configurations.models.blocks.output_blocks import fc_output
from configurations.models.blocks.layers import conv2d_layer, pool_layer, flat, fc_layer
import tensorflow as tf

#%%
def architecture_start(input, add_filters, features):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
  pool2 = pool_layer(conv1, 2, name='pool2')
  conv3 = conv2d_layer(pool2, [5, 5], 128, name='conv3')
  pool_additional_filters = pool_layer(add_filters, 2, name='pool_additional_filters')
  augmented_image = tf.concat(3, [conv3, pool_additional_filters])
  pool4 = pool_layer(augmented_image, 2, name='pool4')
  conv5 = conv2d_layer(pool4, [3, 3], 256, name='conv5')
  return conv5
  
def architecture_end(input, add_filters, features):
  conv6 = conv2d_layer(input, [3, 3], 256, name='conv6')
  conv7 = conv2d_layer(conv6, [3, 3], 128, name='conv7')
  pool8 = pool_layer(conv7, 2, name='pool8')
  print(pool8.get_shape())
  
  flat_features = flat(pool8)
  dropout_features = tf.nn.dropout(features, 0.8)
  full_features = tf.concat(1, [flat_features, dropout_features])
  units = flat_features.get_shape()[1].value
  output_features = fc_layer(input=full_features, units=units, name='output_features')
  return output_features
  
def output(input, dataset):
  out = fc_output(input, dataset.classes, [128], name='output')
  return out
