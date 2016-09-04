# -*- coding: utf-8 -*-
from configurations.models.blocks.output_blocks import fc_output
from configurations.models.blocks.layers import conv2d_layer, pool_layer, flat, fc_layer
from configurations.models.blocks.architecture_blocks import resnet_block

import tensorflow as tf

#%%
def architecture_start(input, add_filters, features):
  print(input.get_shape())
  conv1 = conv2d_layer(input, [11, 11], 96, name='conv1')
  pool2 = pool_layer(conv1, 2, name='pool2')
  res_block_1 = resnet_block(pool2, 'res_block_1')
  res_block_2 = resnet_block(res_block_1, 'res_block_2')
  res_block_3 = resnet_block(res_block_2, 'res_block_3')
  res_block_4 = resnet_block(res_block_3, 'res_block_4')
  pool5 = pool_layer(res_block_4, 2, name='pool5')
  conv6 = conv2d_layer(pool5, [5, 5], 128, name='conv6')
  res_block_7 = resnet_block(conv6, 'res_block_7')
  res_block_8 = resnet_block(res_block_7, 'res_block_8')
  return res_block_8
  
def architecture_end(input, add_filters, features):
  res_block_9 = resnet_block(input, 'inc_res_block_9')
  res_block_10 = resnet_block(res_block_9, 'inc_res_block_10')
  pool11 = pool_layer(res_block_10, 2, name='pool11')
  print(pool11.get_shape())
  
  flat_features = flat(pool11)
  dropout_features = tf.nn.dropout(features, 0.8)
  full_features = tf.concat(1, [flat_features, dropout_features])
  units = 1024
  output_features = fc_layer(input=full_features, units=units, name='output_features')
  return output_features
    
def output(input, dataset):
  out = fc_output(input, dataset.classes, [128], name='output')
  return out


